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

import tensorboard

sys.path.append("13-atdd_package")
from atdd_manager import ATDDManager

# log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
# writer = SummaryWriter(log_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

params = {  # RNN specified
    "bn_layer": "false",
    "act_func": "leaky_relu",
    "grad_clip": "",  # unused
    "init": "kaiming",
    "opt": "sgd",

    "batch_size": 64,
    'lr': 0.01,

    "weight_decay": 0.001,
    "gamma": 0.7,
    "hidden_size": 64,
    "num_layers": 2,

}

manager = ATDDManager()


def set_seed():
    seed = 529
    print("seed: ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def choose_act_func():
    act_func = params["act_func"]
    if act_func == "relu":
        return nn.ReLU()
    elif act_func == "leaky_relu":
        return nn.LeakyReLU()
    elif act_func == "relu6":
        return nn.ReLU6()


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


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=28,
            hidden_size=params["hidden_size"],  # rnn hidden unit
            num_layers=params["num_layers"],  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.bn = nn.BatchNorm1d(params["hidden_size"])
        self.relu = choose_act_func()
        self.out = nn.Linear(params["hidden_size"], 10)

        self.relu_module_name = ["rnn"]  # calc param zero rate
        self.module_name_flow_2dlist = [["rnn", "out"]]  # calc param grad

    def forward(self, x):
        # x batch_size,1,size,size
        x = torch.squeeze(x)
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        if params["bn_layer"] is True:
            out = self.bn(r_out[:, -1, :])
            out = self.out(out)
        else:
            out = self.out(r_out[:, -1, :])
        return out


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
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('../data', train=False, transform=transform)
    train_data, validate_data = torch.utils.data.random_split(train_data, [50000, 10000])
    train_dataloader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    validate_dataloader = torch.utils.data.DataLoader(validate_data, **test_kwargs)

    model = RNN().to(device)
    model = init_model_param(model)

    opt_func = choose_opt_func()
    optimizer = opt_func(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = StepLR(optimizer, step_size=1, gamma=params["gamma"])

    loss_fn = nn.CrossEntropyLoss()
    epochs = params["epoch"] if "epoch" in params else 20

    manager.init_module_basic(model)
    manager.init_cond(optimizer, [train_dataloader, test_dataloader, validate_dataloader], params["lr"])
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
