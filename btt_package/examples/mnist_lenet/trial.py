import random
import sys

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from btt.trial_manager import BttTrialManager
from btt.utils import RecordMode, ObtainMode

# log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
# writer = SummaryWriter(log_dir)

# resource limitaion
# global device, batch_quota, epoch_quota
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_quota = np.inf
epoch_quota = np.inf
print(f"Using {device} device")

params = {
    "conv1_k_num": 6,
    "pool1_size": 3,
    "conv2_k_num": 17,
    "pool2_size": 2,
    "full_num": 84,
    "conv_k_size": 4,
    "lr": 0.1,
    "weight_decay": 0.01,
    "drop_rate": 0.5,
    #
    "batch_norm": 1,
    "drop": 1,
    "batch_size": 300,
    "data_norm": 1,
    "gamma": 0.7,
    "step_size": 1,
    "grad_clip": 1,
    "clip_thresh": 10,
    "act": 0,
    "opt": 0,
    "pool": 0,
    # bn / batch size / transform / gamma / step size XXX / grad clip / opt / pool / momentum XXX
}

seed = 529
manager = None


def clip_gradient(opt, clip_thresh=params["clip_thresh"]):
    for group in opt.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-clip_thresh, clip_thresh)


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
    opt_func_name = params["opt"]
    if opt_func_name == 0:
        return optim.Adam
    elif opt_func_name == 1:
        return optim.SGD
    elif opt_func_name == 2:
        return optim.Adadelta
    elif opt_func_name == 3:
        return optim.Adagrad
    elif opt_func_name == 4:
        return optim.RMSprop


def choose_pool():
    pool_name = params["pool"]
    if pool_name == 0:
        return nn.MaxPool2d
    elif pool_name == 1:
        return nn.AvgPool2d


def set_seed():
    print("seed: ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, params["conv1_k_num"], params["conv_k_size"])
        self.conv2 = nn.Conv2d(params["conv1_k_num"], params["conv2_k_num"], params["conv_k_size"])
        # an affine operation: y = Wx + b
        self.feature_num = self.num_flat_features_()
        self.fc1 = nn.Linear(self.feature_num, params["full_num"])  # 5*5 from image dimension
        self.fc2 = nn.Linear(params["full_num"], params["full_num"])
        self.fc3 = nn.Linear(params["full_num"], 10)
        self.drop = nn.Dropout(params["drop_rate"])
        self.bn1 = nn.BatchNorm2d(params["conv1_k_num"])
        self.bn2 = nn.BatchNorm2d(params["conv2_k_num"])
        self.act = choose_act()
        self.pool1 = choose_pool()(params["pool1_size"], stride=2)
        self.pool2 = choose_pool()(params["pool2_size"], stride=2)

    def forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.bn1(x) if params["batch_norm"] == 1 else x
        x = self.pool2(self.act(self.conv2(x)))
        x = self.bn2(x) if params["batch_norm"] == 1 else x
        x = x.view(-1, self.feature_num)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.drop(x) if params["drop"] == 1 else x
        x = self.fc3(x)
        return x

    def num_flat_features_(self):
        r = 28
        r = r - params["conv_k_size"] + 1
        r = (r - params["pool1_size"]) // 2 + 1
        r = r - params["conv_k_size"] + 1
        r = (r - params["pool2_size"]) // 2 + 1
        return r * r * params["conv2_k_num"]


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    acc_accumulate = 0
    loss_accumulate = 0
    nb_samples = len(dataloader.dataset)
    manager.record_metric({"mode": RecordMode.EpochTrainBegin, "model": model})
    for batch_idx, (X, y) in enumerate(dataloader):
        if batch_idx + 1 >= batch_quota:
            break
        manager.record_metric({"mode": RecordMode.TrainIterBegin})
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_accumulate += float(loss.item()) * X.shape[0]
        # optimizer.zero_grad()
        acc_accumulate += torch.Tensor(pred.argmax(1) == y).sum().item()  # batch sum
        loss.backward()
        if params["grad_clip"] == 1:
            clip_gradient(optimizer)
        optimizer.step()
        ##############################
        manager.record_metric({"mode": RecordMode.TrainIterEnd, "model": model})
        optimizer.zero_grad()
        if batch_idx % (nb_samples // len(X) // 10) == 0:
            loss, current = loss.item(), (batch_idx + 1) * len(X)
            print(f"train_loss: {loss:>7f}  [{current:>5d}/{nb_samples:>5d}]")

    acc = acc_accumulate / nb_samples
    loss = loss_accumulate / nb_samples
    d = {"mode": RecordMode.EpochTrainEnd, "model": model, "train_acc": acc, "train_loss": loss}
    manager.record_metric(d)
    return acc, loss


def standard_test(dataloader, model, loss_fn):
    model.eval()
    loss, acc = 0, 0
    manager.record_metric({"mode": RecordMode.EpochValBegin, "model": model})
    with torch.no_grad():
        for X, y in dataloader:
            manager.record_metric({"mode": RecordMode.ValIterBegin})
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()  # batch ave
            acc += (pred.argmax(1) == y).type(torch.float).sum().item()  # batch sum
            manager.record_metric({"mode": RecordMode.ValIterEnd})
    loss /= len(dataloader)
    acc = acc / len(dataloader.dataset)
    manager.record_metric({"mode": RecordMode.EpochValEnd, "model": model, "val_acc": acc, "val_loss": loss})
    return loss, acc


def test(dataloader, model, loss_fn):
    loss, acc = standard_test(dataloader, model, loss_fn)
    return acc, loss


def main(*args, **kwargs):
    global manager
    manager = BttTrialManager(seed=seed)

    print("experiment_id: ", manager.get_experiment_id())
    print("trial_id: ", manager.get_trial_id())
    optimized_params = manager.get_trial_parameters()
    params.update(optimized_params)
    print("params: ", params)

    train_kwargs = {'batch_size': params["batch_size"]}
    test_kwargs = {'batch_size': params["batch_size"]}
    global device, batch_quota, epoch_quota
    if device == "cuda":
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]) if params["data_norm"] == 1 else transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST('../../../data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('../../../data', train=False, transform=transform)

    train_data, valid_data = torch.utils.data.random_split(train_data, [50000, 10000])
    train_dataloader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, **test_kwargs)

    model = LeNet5().to(device)

    optimizer = choose_opt()(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])
    loss_fn = nn.CrossEntropyLoss()
    global max_nb_epoch, max_nb_batch
    max_nb_epoch = 3  # tset
    max_nb_batch = len(train_dataloader)

    d = {"mode": RecordMode.Begin, "model": model, "max_nb_epoch": max_nb_epoch, "max_nb_batch": max_nb_batch}
    manager.record_metric(d)
    for epoch_idx in range(max_nb_epoch):
        if epoch_idx >= epoch_quota:
            break
        print(f"Epoch {epoch_idx + 1}\n-------------------------------")
        manager.record_metric({"mode": RecordMode.EpochBegin, "model": model, "epoch_idx": epoch_idx})

        # future: + opt + model (all quantized)
        d = {"device": device, "batch_quota": batch_quota, "epoch_quota": epoch_quota}
        d = manager.update_resource_params(d)
        device, batch_quota, epoch_quota = d["device"], d["batch_quota"], d["epoch_quota"]

        train_acc, train_loss = train(train_dataloader, model, loss_fn, optimizer)
        val_acc, val_loss = test(valid_dataloader, model, loss_fn)
        scheduler.step()
        manager.record_metric({"mode": RecordMode.EpochEnd})
        manager.report_intermediate_result()
        print(f"val_loss: {val_loss:>7f}  val_acc: {val_acc:>2f}  ")

    test_acc, test_loss = test(test_dataloader, model, loss_fn)
    manager.record_metric({"mode": RecordMode.End, "test_acc": test_acc, "test_loss": test_loss})

    rule_name = "val_acc"
    d = {"rule_name": rule_name, "mode": ObtainMode.AllWait}
    metric_value = manager.obtain_metric(d)
    print(rule_name, metric_value)

    manager.report_final_result()

    return


if __name__ == '__main__':
    set_seed()
    main()
