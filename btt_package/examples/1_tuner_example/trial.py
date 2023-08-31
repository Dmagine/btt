import random

import numpy as np
import torch
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
        self.conv1_k_size = self.conv2_k_size = 3
        self.conv1_k_num = 16
        self.conv2_k_num = 32
        self.pool1_size = self.pool2_size = 2

        self.conv1 = nn.Conv2d(1, self.conv1_k_num, self.conv1_k_size)
        self.conv2 = nn.Conv2d(self.conv1_k_num, self.conv2_k_num, self.conv2_k_size)
        self.feature_len = self.calc_feature_len()
        self.fc1 = nn.Linear(self.feature_len, 1024)
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

    def calc_feature_len(self):
        r = 28  # mnist
        stride = 1
        r = (r - self.conv1_k_size + 1) // stride
        r = (r - self.pool1_size) // self.pool1_size + 1
        r = (r - self.conv2_k_size + 1) // stride
        r = (r - self.pool2_size) // self.pool2_size + 1
        l = r * r * self.conv2_k_num
        print(l)
        return l


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    acc_accumulate = 0
    loss_accumulate = 0
    nb_samples = len(dataloader.dataset)
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_accumulate += float(loss.item()) * X.shape[0]
        acc_accumulate += torch.Tensor(pred.argmax(1) == y).sum().item()  # batch sum
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % (nb_samples // len(X) // 10) == 0:
            loss, current = loss.item(), (batch_idx + 1) * len(X)
            print(f"train_loss: {loss:>7f}  [{current:>5d}/{nb_samples:>5d}]")

    acc = acc_accumulate / nb_samples
    loss = loss_accumulate / nb_samples
    return acc, loss


def standard_test(dataloader, model, loss_fn):
    model.eval()
    loss, acc = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()  # batch ave
            acc += (pred.argmax(1) == y).type(torch.float).sum().item()  # batch sum
    loss /= len(dataloader)
    acc = acc / len(dataloader.dataset)
    return loss, acc


def test(dataloader, model, loss_fn):
    loss, acc = standard_test(dataloader, model, loss_fn)
    return acc, loss


def trial_func(*args, **kwargs):
    manager = kwargs["manager"]
    batch_size = manager.suggest("batch_size")

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
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('../data', train=False, transform=transform)

    train_data, valid_data = torch.utils.data.random_split(train_data, [50000, 10000])
    train_dataloader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, **test_kwargs)

    model = LeNet5().to(device)

    lr = manager.suggest("lr")
    weight_decay = 1e-4
    step_size = 3
    gamma = 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = nn.CrossEntropyLoss()
    max_nb_epoch = 3  # test

    for epoch_idx in range(max_nb_epoch):
        print(f"Epoch {epoch_idx + 1}\n-------------------------------")
        train_acc, train_loss = train(train_dataloader, model, loss_fn, optimizer)
        val_acc, val_loss = test(valid_dataloader, model, loss_fn)
        scheduler.step()
        print(f"val_loss: {val_loss:>7f}  val_acc: {val_acc:>2f}  ")

    test_acc, test_loss = test(test_dataloader, model, loss_fn)
    print(f"test_loss: {test_loss:>7f}  test_acc: {test_acc:>2f}  ")
    return


if __name__ == '__main__':
    set_seed()
    trial_func()
