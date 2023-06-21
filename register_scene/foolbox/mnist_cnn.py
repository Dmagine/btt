#!/usr/bin/env python3
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from foolbox.models import PyTorchModel
from foolbox.utils import accuracy
from torchvision import transforms, datasets


def create(bounds) -> PyTorchModel:
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Dropout2d(0.5),
        nn.Linear(128, 10),
    )
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mnist_cnn.pth")
    model.load_state_dict(torch.load(path))  # type: ignore
    model.eval()
    # preprocessing = dict(mean=0.1307, std=0.3081)
    fmodel = PyTorchModel(model, bounds=bounds)
    return fmodel


def main():
    # test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {'batch_size': 64}
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
        transforms.Normalize((0.1307,), (0.3081,))  ## (0,1) -> (-??,+??)
    ])
    train_data = datasets.MNIST('../../../data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('../../../data', train=False, transform=transform)

    print(test_data.data.shape, test_data.targets.shape)

    train_data, validate_data = torch.utils.data.random_split(train_data, [50000, 10000])
    train_dataloader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    validate_dataloader = torch.utils.data.DataLoader(validate_data, **test_kwargs)

    # print(accuracy(fmodel, images, labels))

    # images, labels = next(iter(validate_dataloader))
    images = torch.cat([images for images, labels in validate_dataloader], dim=0)
    labels = torch.cat([labels for images, labels in validate_dataloader], dim=0)
    images, labels = images.to(device), labels.to(device)
    print(images.shape, labels.shape, torch.min(images), torch.max(images), torch.median(images))
    fmodel = create(bounds=(torch.min(images), torch.max(images)))
    # images, labels = samples(fmodel, dataset="mnist", batchsize=20)
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")
    import foolbox.attacks as fa
    attacks = [
        fa.FGSM(),
        fa.LinfPGD(),
        fa.LinfBasicIterativeAttack(),
        fa.LinfAdditiveUniformNoiseAttack(),
        fa.LinfDeepFoolAttack(),
    ]
    epsilons = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.1, 0.3, 0.5, 1.0, ]
    print("epsilons")
    print(epsilons)
    print("")
    # apply the attack
    attack_success = np.zeros((len(attacks), len(epsilons), len(images)), dtype=bool)
    for i, attack in enumerate(attacks):
        print(images.shape, labels.shape)
        _, _, success = attack(fmodel, images, labels, epsilons=epsilons)
        success = success.cpu()
        assert success.shape == (len(epsilons), len(images))
        success_ = success.numpy()
        assert success_.dtype == bool
        attack_success[i] = success_
        print(attack)
        print("  ", 1.0 - success_.mean(axis=-1).round(2))

    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked) using the best attack per sample
    robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
    print("")
    print("-" * 79)
    print("")
    print("worst case (best attack per-sample)")
    print("  ", robust_accuracy.round(2))
    print("")

    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")


if __name__ == "__main__":
    import logging

    # 配置日志记录器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个文件处理器
    f_name = __file__.split('/')[-1].split('.')[0]
    log_f_name = "./logs/" + f_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"

    file_handler = logging.FileHandler(log_f_name)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 创建一个流处理器（命令行输出）
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)

    # 将处理器添加到日志记录器
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # 记录日志
    logger.info("This is an info message")
    logger.warning("This is a warning message")

    main()

    # 关闭日志记录器
    logger.removeHandler(file_handler)
    logger.removeHandler(stream_handler)
    file_handler.close()
