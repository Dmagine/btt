import random
import sys

import nni
import numpy as np
import torch
import torch.optim as optim
import torchvision.models
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
    "layer_num": 18,
    "batch_size": 90,
    "lr": 0.1,
    "weight_decay": 0.01,
    "gamma": 0.7,
    "opt": 0
}

seed = 529
manager = ATDDManager(seed=seed)


def clip_gradient(opt, clip_thresh=10):
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


def set_seed():
    print("seed: ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


def choose_net():
    if params["layer_num"] == 18:
        return torchvision.models.resnet18(pretrained=False)
    elif params["layer_num"] == 34:
        return torchvision.models.resnet34(pretrained=False)
    elif params["layer_num"] == 50:
        return torchvision.models.resnet50(pretrained=False)
    elif params["layer_num"] == 101:
        return torchvision.models.resnet101(pretrained=False)
    elif params["layer_num"] == 152:
        return torchvision.models.resnet152(pretrained=False)


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
    train_data = datasets.CIFAR10('../../../data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('../../../data', train=False, transform=transform_test)

    train_data, validate_data = torch.utils.data.random_split(train_data, [40000, 10000])
    train_dataloader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    validate_dataloader = torch.utils.data.DataLoader(validate_data, **test_kwargs)

    model = choose_net().to(device)

    optimizer = choose_opt()(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = StepLR(optimizer, step_size=1, gamma=params["gamma"])
    loss_fn = nn.CrossEntropyLoss()
    epochs = 15
    epochs = epochs if 'TRIAL_BUDGET' not in params else params['TRIAL_BUDGET']  # BOHB

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
    return


if __name__ == '__main__':
    set_seed()
    main()
    # model = torchvision.models.resnet18(pretrained=False)
    # for (module_name, module) in model.named_modules():
    #     if type(module) in [nn.Conv2d, nn.Linear, nn.LSTM, nn.RNN]:
    #         for (param_name, param) in module.named_parameters():
    #             if "weight" == param_name:
    #                 print(module_name, param_name, param.shape)

