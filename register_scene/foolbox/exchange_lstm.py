from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExchangeDataset(Dataset):
    def __init__(self, seq_len, target_len, mode="train"):
        self.split_rate = 0.8
        self.seq_len = seq_len
        self.target_len = target_len
        data_path = "../../../data/dataset/exchange_rate/exchange_rate.csv"
        data = pd.read_csv(data_path, header=None).values[:, 1:-1].astype("float32")
        data = torch.tensor(data).to(device)
        print(data.shape)
        if mode == "train":
            self.data = data[:int(self.split_rate * len(data))]
        else:
            self.data = data[int(self.split_rate * len(data)):]

    def __len__(self):
        return len(self.data) - self.seq_len - self.target_len + 1

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]
        pred = self.data[idx + self.seq_len:idx + self.seq_len + self.target_len]
        return seq, pred


class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, target_len):
        super(LstmModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_len = target_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, ):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)  # 只取最后一个时间步的输出作为预测结果 no
        return out


def main():
    seq_len = 96
    label_len = 48
    pred_len = 96
    train_dataset = ExchangeDataset(mode="train", seq_len=seq_len, target_len=label_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = ExchangeDataset(mode="test", seq_len=seq_len, target_len=pred_len)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = LstmModel(input_size=7, hidden_size=128, num_layers=2, output_size=7, target_len=label_len).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss_list, val_loss_list = [], []
    print("====================================")
    for epoch in range(50):
        if epoch % 10 == 0:
            print("epoch:{}".format(epoch))
        train_loss = 0
        for seq, pred in train_loader:
            # -》 (batch_size, seq_length, num_features)
            # print(seq.shape, pred.shape)
            out = model(seq)[:, -label_len:]
            loss = loss_fn(out, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_list.append(train_loss / len(train_loader))
        print("epoch:{},loss:{}".format(epoch, train_loss_list[-1]))
        val_loss = 0
        for seq, pred in val_loader:
            out = model(seq)[:, -pred_len:]
            loss = loss_fn(out, pred)
            val_loss += loss.item()
        val_loss_list.append(val_loss / len(val_loader))
        print("val_loss:{}".format(val_loss_list[-1]))
        print("====================================")
        import foolbox
        import foolbox.attacks as fa
        val_seqs = torch.cat([seq for seq, _ in val_loader], dim=0)
        val_preds = torch.cat([pred.reshape(pred.shape[0], -1) for _, pred in val_loader], dim=0)
        val_bounds = (torch.min(val_seqs), torch.max(val_seqs))

        class Net(nn.Module):
            def __init__(self, model, pred_len):
                super(Net, self).__init__()
                self.model = model
                self.pred_len = pred_len

            def forward(self, x):
                r = self.model(x)[:, -pred_len:]
                return r.reshape(r.shape[0], -1)

        foolbox.criteria.Misclassification()
        fmodel = foolbox.models.PyTorchModel(model, bounds=val_bounds, device=device)
        attacks = [
            fa.FGSM(),
            fa.LinfPGD(),
            fa.LinfBasicIterativeAttack(),
            fa.LinfAdditiveUniformNoiseAttack(),
            fa.LinfDeepFoolAttack(),
        ]
        epsilons = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.1, 0.3, 0.5, 1.0]
        attack_success = np.zeros((len(attacks), len(epsilons), len(val_seqs)), dtype=bool)
        for i, attack in enumerate(attacks):
            print(val_seqs.shape, val_preds.shape)
            _, _, success = attack(fmodel, val_seqs, val_preds, epsilons=epsilons)
            assert success.shape == (len(epsilons), len(val_seqs))
            success_ = success.numpy()
            assert success_.dtype == bool
            attack_success[i] = success_
            print(attack, "  ", 1.0 - success_.mean(axis=-1).round(2))
        robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
        print("\n", "-" * 79, "\n")
        print("worst case (best attack per-sample)")
        print("  ", robust_accuracy.round(2), "\n")
        print("robust accuracy for perturbations with")
        for eps, acc in zip(epsilons, robust_accuracy):
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        print("====================================")
        # plot: x->epoch y->val_loss
    plt.plot(range(len(val_loss_list)), val_loss_list)
    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["val_loss", "train_loss"])
    plt.show()


if __name__ == "__main__":
    import logging

    # 配置日志记录器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个文件处理器
    f_name = __file__.split('/')[-1].split('.')[0]
    log_f_name = "./logs/" + f_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"

    file_handler = logging.FileHandler(log_f_name)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 创建一个流处理器（命令行输出）
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)

    # 将处理器添加到日志记录器
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # 记录日志
    logger.info("This is an info message")
    logger.warning("This is a warning message")

    main()

    # 关闭日志记录器
    logger.removeHandler(file_handler)
    logger.removeHandler(stream_handler)
    file_handler.close()
