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

sys.path.append("../../atdd_package")
from atdd_manager import ATDDManager

# log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
# writer = SummaryWriter(log_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
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
manager = ATDDManager(seed=seed)


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

    # elif pool_name == 2:
    #     return nn.AdaptiveMaxPool2d
    # elif pool_name == 3:
    #     return nn.AdaptiveAvgPool2d


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
        if params["grad_clip"] == 1:
            clip_gradient(optimizer)
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


def save_checkpoint(model, checkpoint_path):
    torch.save(model.state_dict(), checkpoint_path)


def load_checkpoint(checkpoint_path):
    model_state_dict = torch.load(checkpoint_path)
    return model_state_dict


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

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    # train_data = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)  ###
    # test_data = datasets.FashionMNIST('../data', train=False, transform=transform)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]) if params["data_norm"] == 1 else transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST('../../../data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('../../../data', train=False, transform=transform)

    train_data, validate_data = torch.utils.data.random_split(train_data, [50000, 10000])
    train_dataloader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    validate_dataloader = torch.utils.data.DataLoader(validate_data, **test_kwargs)

    model = LeNet5().to(device)
    # all_checkpoint_dir = os.getenv('NNI_CHECKPOINT_DIRECTORY')
    # if all_checkpoint_dir is None:
    #     all_checkpoint_dir = os.path.join(os.getenv('NNI_SYS_DIR'), "../../checkpoint")  # NNI_SYS_DIR
    # save_checkpoint_path = os.path.join(params['save_checkpoint_dir'], 'model.pth') \
    #     if "save_checkpoint_dir" in params else None
    # load_checkpoint_path = os.path.join(params['load_checkpoint_dir'], 'model.pth') \
    #     if "load_checkpoint_dir" in params else None
    # print(all_checkpoint_dir, save_checkpoint_path, load_checkpoint_path)
    # if load_checkpoint_path is not None and os.path.isfile(load_checkpoint_path):
    #     model_state_dict = load_checkpoint(load_checkpoint_path)
    #     print("load : ", load_checkpoint_path)
    #     model.load_state_dict(model_state_dict)

    optimizer = choose_opt()(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])
    loss_fn = nn.CrossEntropyLoss()
    epochs = 15
    epochs = epochs if 'TRIAL_BUDGET' not in params else params['TRIAL_BUDGET']  # BOHB
    # epochs = epochs if 'save_checkpoint_dir' not in params else 1  # PBT

    manager.init_basic(model, train_dataloader)
    # manager.init_cond(optimizer, [train_dataloader, test_dataloader, validate_dataloader], params["lr"])
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
    # if "save_checkpoint_dir" in params and not os.path.exists(params['save_checkpoint_dir']):
    #     os.makedirs(params['save_checkpoint_dir'])
    # print("save : ", save_checkpoint_path)
    # save_checkpoint(model, save_checkpoint_path)
    return


if __name__ == '__main__':
    set_seed()
    main()
