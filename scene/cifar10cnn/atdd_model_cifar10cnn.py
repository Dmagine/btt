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

sys.path.append("../../atdd_package")
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
    "mlp_f_num": 200,
    "conv_k_size": 2,
    "lr": 0.1,
    "weight_decay": 0.1,
    "act": 0,
    "opt": 0,
    "drop_rate": 0.5,
    "batch_norm": 1,
    "batch_size": 90
}

seed = 529
manager = ATDDManager(seed=seed)


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
        self.fc1 = nn.Linear(self.feature_num, params["mlp_f_num"])  # 5*5 from image dimension
        self.fc2 = nn.Linear(params["mlp_f_num"], params["mlp_f_num"])
        self.fc3 = nn.Linear(params["mlp_f_num"], 10)
        self.drop = nn.Dropout(params["drop_rate"])
        self.act = choose_act()
        self.bn1 = nn.BatchNorm2d(params["conv1_k_num"])
        self.bn2 = nn.BatchNorm2d(params["conv2_k_num"])
        self.bn3 = nn.BatchNorm2d(params["conv3_k_num"])
        self.bn4 = nn.BatchNorm2d(params["conv4_k_num"])

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.bn1(x) if params["batch_norm"] == 1 else x
        x = F.max_pool2d(self.act(self.conv2(x)), 2)
        x = self.bn2(x) if params["batch_norm"] == 1 else x
        x = self.act(self.conv3(x))
        x = self.bn3(x) if params["batch_norm"] == 1 else x
        x = F.max_pool2d(self.act(self.conv4(x)), 2)
        x = self.bn4(x) if params["batch_norm"] == 1 else x
        x = x.view(-1, self.feature_num)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

    def num_flat_features_(self):
        r = 32  # cifar10
        r = r - params["conv_k_size"] + 1
        r = r - params["conv_k_size"] + 1
        r = (r - 2) // 2 + 1
        r = r - params["conv_k_size"] + 1
        r = r - params["conv_k_size"] + 1
        r = (r - 2) // 2 + 1
        return r * r * params["conv4_k_num"]


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    correct = 0
    loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss += float(loss.item())
        optimizer.zero_grad()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss.backward()
        manager.collect_in_training(model)
        optimizer.step()
    acc = correct / size
    loss_ave = loss / num_batches
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

    model = CNN().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
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