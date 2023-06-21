#!/usr/bin/env python3
from datetime import datetime

import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np


def main():
    # instantiate a model (could also be a TensorFlow or JAX model)
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")
    print("")

    attacks = [
        fa.FGSM(),
        fa.LinfPGD(),
        fa.LinfBasicIterativeAttack(),
        fa.LinfAdditiveUniformNoiseAttack(),
        fa.LinfDeepFoolAttack(),
    ]

    epsilons = [
        0.0,
        0.0005,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.005,
        0.01,
        0.02,
        0.03,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    print("epsilons")
    print(epsilons)
    print("")

    attack_success = np.zeros((len(attacks), len(epsilons), len(images)), dtype=bool)
    for i, attack in enumerate(attacks):
        _, _, success = attack(fmodel, images, labels, epsilons=epsilons)
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