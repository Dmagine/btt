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
import torch.nn.functional as F

sys.path.append("atdd_package")
from atdd_manager import ATDDManager

# log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
# writer = SummaryWriter(log_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

params = {
    "conv1_k_num": 8,
    "conv2_k_num": 32,
    "conv3_k_num": 64,
    "conv4_k_num": 64,
    "full_num": 84,
    "conv_k_size": 2,
    "lr": 0.1,
    "l2_factor": 0.01,
    "act": "relu",
    "reg": "dropout"
}

seed = 529
manager = ATDDManager(seed=seed)

def choose_act():
    act_func = params["act"]
    if act_func == "relu":
        return nn.ReLU()
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func == "elu":
        return nn.ELU()

def set_seed():
    print("seed: ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, params["conv1_k_num"], params["conv_k_size"])
        self.conv2 = nn.Conv2d(params["conv1_k_num"], params["conv2_k_num"], params["conv_k_size"])
        self.conv3 = nn.Conv2d(params["conv2_k_num"], params["conv3_k_num"], params["conv_k_size"])
        self.conv4 = nn.Conv2d(params["conv3_k_num"], params["conv4_k_num"], params["conv_k_size"])
        # an affine operation: y = Wx + b
        self.feature_num = self.num_flat_features_()
        self.fc1 = nn.Linear(self.feature_num, params["full_num"])  # 5*5 from image dimension
        self.fc2 = nn.Linear(params["full_num"], 100)
        self.drop = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(params["full_num"])
        self.act = choose_act()

    def forward(self, x):
        x = F.max_pool2d(self.act(self.conv1(x)), 2, 2)
        x = F.max_pool2d(self.act(self.conv2(x)), 2, 2)
        x = F.max_pool2d(self.act(self.conv3(x)), 2, 2)
        x = F.max_pool2d(self.act(self.conv4(x)), 2, 2)
        x = x.view(-1, self.feature_num)
        x = self.act(self.fc1(x))
        if params["reg"] == "dropout":
            x = self.drop(x)
        elif params["reg"] == "batchnorm":
            x = self.bn(x)
        x = self.fc2(x)
        return x

    def num_flat_features_(self):
        r = 224
        for i in range(4):
            r = r - params["conv_k_size"] + 1
            r = (r - 2) // 2 + 1
        return r * r * params["conv4_k_num"]


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    loss_accumulate = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_accumulate += float(loss.item()) * batch
        optimizer.zero_grad()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss.backward()
        manager.collect_in_training(model)
        optimizer.step()
    acc = correct / size
    loss_ave = loss_accumulate / size
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
    print("experiment_id: ", nni.get_experiment_id())
    print("trial_id: ", nni.get_trial_id())
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print("params: ", params)

    train_kwargs = {'batch_size': 256}
    test_kwargs = {'batch_size': 256}
    if device == "cuda":
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.Resize(256),  # transforms.Scale(256)
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #### CIFAR100 dataset
    train_data = datasets.CIFAR100('../data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100('../data', train=False, transform=transform)
    train_data, validate_data = torch.utils.data.random_split(train_data, [40000, 10000])
    train_dataloader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    validate_dataloader = torch.utils.data.DataLoader(validate_data, **test_kwargs)

    model = CNN().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=params["lr"], weight_decay=params["l2_factor"])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 50

    manager.init_module_basic(model)
    manager.init_cond(optimizer, [train_dataloader, test_dataloader, validate_dataloader], params["lr"])
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
