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

sys.path.append("atdd_package")
from atdd_manager import ATDDManager

# log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
# writer = SummaryWriter(log_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

params = {  # RNN specified
    "grad_clip": True,
    "init": "kaiming",
    "opt": "sgd",

    "batch_size": 64,
    'lr': 0.01,

    "weight_decay": 0.001,
    "gamma": 0.7,
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


def init_model_param(m: nn.Module):
    init_func_name = params["init"]
    if init_func_name == "none":
        return m
    for (module_name, module) in m.named_modules():
        if type(module) in [nn.Conv2d, nn.Linear, nn.LSTM]:
            if type(module) in [nn.Conv2d]:
                if init_func_name == "xvaier":
                    nn.init.xavier_normal(module.weight)
                elif init_func_name == "kaiming":
                    nn.init.kaiming_normal(module.weight)
            if type(module) in [nn.Linear]:
                if init_func_name == "xvaier":
                    nn.init.xavier_uniform(module.weight)
                elif init_func_name == "kaiming":
                    nn.init.kaiming_uniform(module.weight)
            if type(module) in [nn.LSTM]:
                if init_func_name == "xvaier":
                    for i in range(params["num_layers"]):
                        nn.init.xavier_uniform(getattr(module, "weight_ih_l" + str(i)))
                        nn.init.xavier_uniform(getattr(module, "weight_hh_l" + str(i)))
                elif init_func_name == "kaiming":
                    for i in range(params["num_layers"]):
                        nn.init.kaiming_uniform(getattr(module, "weight_ih_l" + str(i)))
                        nn.init.kaiming_uniform(getattr(module, "weight_hh_l" + str(i)))
    return m


def choose_opt_func():
    opt_func_name = params["opt"]
    if opt_func_name == "adam":
        return optim.Adam
    elif opt_func_name == "sgd":
        return optim.SGD
    elif opt_func_name == "adadelta":
        return optim.Adadelta


class MyLeNet5(nn.Module):
    def __init__(self):
        super(MyLeNet5, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.Sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=5)

        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(128, 84)
        self.bn = nn.BatchNorm1d(84)
        self.output = nn.Linear(84, 10)

        self.relu_pre_module_name = ["c1", "c3"]  # calc paaram grad zero rate
        self.module_name_flow_2dlist = [["c1", "c3", "c5", "f6", "output"]]  # calc param grad

    def forward(self, x):
        # x输入为32*32*1， 输出为28*28*6
        x = self.Sigmoid(self.c1(x))
        # x输入为28*28*6， 输出为14*14*6
        x = self.s2(x)
        # x输入为14*14*6， 输出为10*10*16
        x = self.Sigmoid(self.c3(x))
        # x输入为10*10*16， 输出为5*5*16
        x = self.s4(x)
        # x输入为5*5*16， 输出为1*1*120
        x = self.c5(x)
        x = self.flatten(x)
        # x输入为120， 输出为84
        x = self.f6(x)

        # x输入为84， 输出为10
        x = self.output(x)
        return x


def clip_gradient(opt, grad_clip=10):
    for group in opt.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


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
        if params["grad_clip"] is False:
            clip_gradient(optimizer)
        optimizer.step()
    acc = correct / size
    loss_ave = loss_accumulate / size
    manager.collect_after_training(acc, loss_ave)
    manager.calculate_metrics_after_training()
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)  ###
    test_data = datasets.FashionMNIST('../data', train=False, transform=transform)
    train_data, validate_data = torch.utils.data.random_split(train_data, [50000, 10000])
    train_dataloader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    validate_dataloader = torch.utils.data.DataLoader(validate_data, **test_kwargs)

    model = MyLeNet5().to(device)
    model = init_model_param(model)

    opt_func = choose_opt_func()
    optimizer = opt_func(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = StepLR(optimizer, step_size=1, gamma=params["gamma"])

    loss_fn = nn.CrossEntropyLoss()
    epochs = params["epoch"] if "epoch" in params else 20

    manager.init_basic(model,train_dataloader)
    # manager.init_cond(optimizer, [train_dataloader, test_dataloader, validate_dataloader], params["lr"])
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        manager.refresh_before_epoch_start()

        train(train_dataloader, model, loss_fn, optimizer)
        acc, _ = validate(validate_dataloader, model, loss_fn)
        scheduler.step()
        manager.report_intermediate_result(acc)

        if manager.if_atdd_send_stop():
            break
    acc, _ = test(test_dataloader, model, loss_fn)
    manager.report_final_result(acc)
    return


if __name__ == '__main__':
    set_seed()
    main()
