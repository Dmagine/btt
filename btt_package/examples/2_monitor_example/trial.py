import random

import btt
import numpy as np
import torch
from btt.trial_manager import BttTrialManager
from btt.utils import RecordMode
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# resource limitaion
# global device, batch_quota, epoch_quota
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

seed = 529


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
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(self.feature_num, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)
        self.drop = nn.Dropout(p=0.5)
        self.act = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x


def train(dataloader, model, loss_fn, optimizer, manager):
    model.train()
    acc_accumulate = 0
    loss_accumulate = 0
    nb_samples = len(dataloader.dataset)
    manager.record_metric({"mode": RecordMode.EpochTrainBegin})
    for batch_idx, (X, y) in enumerate(dataloader):
        manager.record_metric({"mode": RecordMode.TrainIterBegin})
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_accumulate += float(loss.item()) * X.shape[0]
        acc_accumulate += torch.Tensor(pred.argmax(1) == y).sum().item()  # batch sum
        loss.backward()
        optimizer.step()
        manager.record_metric({"mode": RecordMode.TrainIterEnd})
        optimizer.zero_grad()
        if batch_idx % (nb_samples // len(X) // 10) == 0:
            loss, current = loss.item(), (batch_idx + 1) * len(X)
            print(f"train_loss: {loss:>7f}  [{current:>5d}/{nb_samples:>5d}]")

    acc = acc_accumulate / nb_samples
    loss = loss_accumulate / nb_samples
    manager.record_metric({"mode": RecordMode.EpochTrainEnd})
    return acc, loss


def standard_test(dataloader, model, loss_fn, manager=None):
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


def trial_func(manager: btt.trial_manager, *args, **kwargs):
    batch_size = manager.suggest_param("batch_size", 64)
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    global device
    if device == "cuda":
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('../../../data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('../../../data', train=False, transform=transform)

    train_data, valid_data = torch.utils.data.random_split(train_data, [50000, 10000])
    train_dataloader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, **test_kwargs)

    model = LeNet5().to(device)

    lr = manager.suggest_param("lr", 1e-3)
    weight_decay = manager.suggest_param("weight_decay", 0.0)
    step_size = manager.suggest_param("step_size", 3)
    gamma = manager.suggest_param("gamma", 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = nn.CrossEntropyLoss()
    max_nb_epoch = 5  # test

    manager.record_metric(d={"mode": RecordMode.Begin})
    for epoch_idx in range(max_nb_epoch):
        print(f"Epoch {epoch_idx + 1}\n-------------------------------")
        manager.record_metric({"mode": RecordMode.EpochBegin, "model": model, "epoch_idx": epoch_idx})

        # future: + opt + model (all quantized)

        train_acc, train_loss = train(train_dataloader, model, loss_fn, optimizer)
        val_acc, val_loss = test(valid_dataloader, model, loss_fn)
        scheduler.step()
        manager.record_metric({"mode": RecordMode.EpochEnd})
        print(f"val_loss: {val_loss:>7f}  val_acc: {val_acc:>2f}  ")

    test_acc, test_loss = test(test_dataloader, model, loss_fn)
    print(f"test_loss: {test_loss:>7f}  test_acc: {test_acc:>2f}  ")
    manager.record_metric({"mode": RecordMode.End})
    return


if __name__ == '__main__':
    set_seed()
    trial_func(None)
