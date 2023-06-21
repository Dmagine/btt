import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss_list, val_loss_list = [], []
    for epoch in range(50):
        logger.info("====================================")
        if epoch % 10 == 0:
            logger.info("epoch:{}".format(epoch))
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
        logger.info("epoch:{},loss:{}".format(epoch, train_loss_list[-1]))
        val_loss = 0
        for seq, pred in val_loader:
            out = model(seq)[:, -pred_len:]
            loss = loss_fn(out, pred)
            val_loss += loss.item()
        val_loss_list.append(val_loss / len(val_loader))
        logger.info("epoch:{},val_loss:{}".format(epoch, val_loss_list[-1]))

        from cleverhans.torch.attacks.noise import noise
        from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
        from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
        # attacks = [fast_gradient_method, projected_gradient_descent, carlini_wagner_l2, spsa, hop_skip_jump_attack,
        #              noise, sparse_l1_descent, semantic]
        # eps_list = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.1, 0.3, 0.5, 1.0]
        eps = 0.01
        nb_batches = 0
        loss, loss_fgm, loss_pgd, loss_spsa, loss_hsja, loss_noise, loss_sld, loss_sem = 0, 0, 0, 0, 0, 0, 0, 0
        x_max = train_dataset.data.max().float()
        x_min = train_dataset.data.min().float()
        logger.info("eps:{}".format(eps))
        logger.info("x_max:{},x_min:{}".format(x_max, x_min))
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)  # clip存在问题 内置要求 numpy 所以用clip只能cpu
            nb_batches += 1

            y_pred = model(x)[:, -pred_len:]
            loss += loss_fn(y_pred, y).item()

            x_fgm = fast_gradient_method(model, x, eps, np.inf)
            y_pred_fgm = model(x_fgm)[:, -pred_len:]
            loss_fgm += loss_fn(y_pred_fgm, y).item()

            x_pgd = projected_gradient_descent(model, x, eps, 0.01, 40, np.inf)
            y_pred_pgd = model(x_pgd)[:, -pred_len:]
            loss_pgd += loss_fn(y_pred_pgd, y).item()

            # x_cwl = carlini_wagner_l2(model, x, y, 0.01, 40, 0.01, 0.9, np.inf) # n_classes
            # y_pred_cwl = model(x_cwl)[:, -pred_len:]
            # loss_cwl += loss_fn(y_pred_cwl, y).item()
            # x_spsa = spsa(model, x, eps, 40,clip_min=x_min, clip_max=x_max) # RuntimeError: expand(torch.cuda.LongTensor{[1, 7]}, size=[256]): the number of sizes provided (1) must be greater or equal to the number of dimensions in the tensor (2)
            # y_pred_spsa = model(x_spsa)[:, -pred_len:]
            # loss_spsa += loss_fn(y_pred_spsa, y).item()
            # x_hsja = hop_skip_jump_attack(model, x, np.inf, clip_min=x_min, clip_max=x_max) # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
            # y_pred_hsja = model(x_hsja)[:, -pred_len:]
            # loss_hsja += loss_fn(y_pred_hsja, y).item()

            x_noise = noise(x, eps, np.inf, clip_min=x_min, clip_max=x_max)
            y_pred_noise = model(x_noise)[:, -pred_len:]
            loss_noise += loss_fn(y_pred_noise, y).item()

            # x_sld = sparse_l1_descent(model, x, eps, eps_iter=eps / 10) # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            # y_pred_sld = model(x_sld)[:, -pred_len:]
            # loss_sld += loss_fn(y_pred_sld, y).item()
            # x_sem = semantic(x, max_val=x_max) # image centered
            # y_pred_sem = model(x_sem)[:, -pred_len:]
            # loss_sem += loss_fn(y_pred_sem, y).item()

        logger.info("val loss on clean examples : {}".format(loss / len(val_loader)))
        logger.info("val loss on FGM adversarial examples : {}".format(loss_fgm / len(val_loader)))
        logger.info("val loss on PGD adversarial examples : {}".format(loss_pgd / len(val_loader)))
        # logger.info("val loss on SPSA adversarial examples : {}".format(loss_spsa / len(val_loader)))
        # logger.info("val loss on HSJA adversarial examples : {}".format(loss_hsja / len(val_loader)))
        logger.info("val loss on NOISE adversarial examples : {}".format(loss_noise / len(val_loader)))
        # logger.info("val loss on SLD adversarial examples : {}".format(loss_sld / len(val_loader)))
        # logger.info("val loss on SEM adversarial examples : {}".format(loss_sem / len(val_loader)))


if __name__ == "__main__":
    f_name = __file__.split('/')[-1].split('.')[0]
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    # stdout_f_name = "./logs/" + f_name + "_" + time_str + "_stdout.txt"
    # stderr_f_name = "./logs/" + f_name + "_" + time_str + "_stderr.txt"
    log_f_name = "./logs/" + f_name + "_" + time_str + "_log.txt"

    import logging

    # 配置日志记录器
    format_str = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=format_str, filename=log_f_name)

    # 创建一个流处理器（命令行输出）
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter(format_str)
    stream_handler.setFormatter(stream_formatter)

    # 将处理器添加到日志记录器
    logger = logging.getLogger()
    logger.addHandler(stream_handler)

    # 记录日志
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    main()

    # 关闭日志记录器
    logger.removeHandler(stream_handler)
    stream_handler.close()
