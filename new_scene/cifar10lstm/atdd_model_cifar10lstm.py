import random
import sys

import nni
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append("../../new_package")
from atdd_manager import ATDDManager

# log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
# writer = SummaryWriter(log_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

seed = 529
params = {
    "hidden_size": 32,
    "mlp_f_num": 200,
    "lr": 0.1,
    "weight_decay": 0.1,
    "act": 0,
    "opt": 0,
    "drop_rate": 0.5,
    "batch_size": 90
}
manager = None


def set_seed():
    print("seed: ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def choose_act():
    act_func = params["act"]
    if act_func == 0:
        return nn.ReLU()
    elif act_func == 1:
        return nn.Tanh()
    elif act_func == 2:
        return nn.Sigmoid()
    elif act_func == 3:
        return nn.ELU()
    elif act_func == 4:
        return nn.LeakyReLU()


def choose_opt():
    opt_func = params["opt"]
    if opt_func == 0:
        return optim.SGD
    elif opt_func == 1:
        return optim.Adam
    elif opt_func == 2:
        return optim.RMSprop
    elif opt_func == 3:
        return optim.Adagrad
    elif opt_func == 4:
        return optim.Adadelta


class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()

        self.lstm1 = nn.LSTM(32, params["hidden_size"], batch_first=True)
        self.lstm2 = nn.LSTM(params["hidden_size"], params["hidden_size"], batch_first=True)
        self.relu = choose_act()
        self.mlp1 = nn.Linear(params["hidden_size"], params["mlp_f_num"])
        self.mlp2 = nn.Linear(params["mlp_f_num"], 10)
        self.dropout = nn.Dropout(params["drop_rate"])

    def forward(self, x):
        def zero_init_hidden():
            return torch.zeros(1, x.size(0), params["hidden_size"]).to(device)
        # x batch_size,n=3,size,size
        # x = torch.squeeze(x)
        # 合并第二和第三个维度
        x = x.view(x.size(0), -1, x.size(-1))
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        x, (h_n, h_c) = self.lstm1(x, (zero_init_hidden(), zero_init_hidden()))
        x, (h_n, h_c) = self.lstm2(x, (h_n, h_c))

        # choose r_out at the last time step
        x = x[:, -1, :]
        x = self.relu(self.mlp1(x))
        x = self.dropout(x)
        x = self.mlp2(x)
        return x


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    correct = 0
    loss_sum = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_sum += float(loss.item())
        optimizer.zero_grad()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss.backward()
        manager.collect_in_training(model)
        optimizer.step()
    acc = correct / size
    loss_ave = loss_sum / num_batches
    manager.collect_after_training(acc, loss_ave)
    manager.calculate_after_training()
    return acc, loss_ave


def standard_test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    acc = correct / size
    return loss, acc


def test(dataloader, model, loss_fn):
    loss, acc = standard_test(dataloader, model, loss_fn)
    manager.collect_after_testing(acc, loss)
    return acc, loss


def validate(dataloader, model, loss_fn):
    loss, acc = standard_test(dataloader, model, loss_fn)
    manager.collect_after_validating(acc, loss)
    return acc, loss


def main():
    global manager, params
    manager = ATDDManager(seed=seed)
    print("experiment_id: ", nni.get_experiment_id())
    print("trial_id: ", nni.get_trial_id())
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print("params: ", params)

    train_kwargs = {'batch_size': params["batch_size"]}
    test_kwargs = {'batch_size': params["batch_size"]}
    if device == "cuda":
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    train_data = datasets.CIFAR10('../../../data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('../../../data', train=False, transform=transform_test)
    train_data, validate_data = torch.utils.data.random_split(train_data, [40000, 10000])
    train_dataloader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    validate_dataloader = torch.utils.data.DataLoader(validate_data, **test_kwargs)

    model = MyLSTM().to(device)

    optimizer = choose_opt()(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 20

    manager.init_basic(model, train_dataloader)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        manager.refresh_before_epoch_start()

        train(train_dataloader, model, loss_fn, optimizer)
        acc, _ = validate(validate_dataloader, model, loss_fn)
        manager.report_intermediate_result(acc)
        scheduler.step()

        if manager.if_atdd_send_stop():
            break
    acc, _ = test(test_dataloader, model, loss_fn)
    manager.report_final_result(acc)
    return


if __name__ == '__main__':
    set_seed()
    main()
