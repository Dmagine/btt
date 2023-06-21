#!/usr/bin/env python3
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    val_loader = torch.utils.data.DataLoader(validate_data, **test_kwargs)

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
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("====================================")
    for epoch in range(50):
        if epoch % 10 == 0:
            print("epoch:{}".format(epoch))
        train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = loss_fn(out, labels)
            acc = (out.argmax(dim=1) == labels).float().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
            train_acc_list.append(acc.item())
        print(len(train_loader))
        print("epoch:{} train_loss:{} train_acc:{}".format(epoch, np.mean(train_loss_list), np.mean(train_acc_list)))
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = loss_fn(out, labels)
            acc = (out.argmax(dim=1) == labels).float().mean()
            val_loss_list.append(loss.item())
            val_acc_list.append(acc.item())
        print("val_loss:{} val_acc:{}".format(np.mean(val_loss_list), np.mean(val_acc_list)))
        from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
        from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
        eps = 0.3
        correct, correct_fgm, correct_pgd, nb_test = 0, 0, 0, 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            x_fgm = fast_gradient_method(model, x, eps, np.inf)
            x_pgd = projected_gradient_descent(model, x, eps, 0.01, 40, np.inf)
            _, y_pred = model(x).max(1)  # model prediction on clean examples
            # print(_.shape, y_pred.shape)
            _, y_pred_fgm = model(x_fgm).max(1)  # model prediction on FGM adversarial examples
            _, y_pred_pgd = model(x_pgd).max(1)  # model prediction on PGD adversarial examples
            nb_test += y.size(0)
            correct += y_pred.eq(y).sum().item()
            correct_fgm += y_pred_fgm.eq(y).sum().item()
            correct_pgd += y_pred_pgd.eq(y).sum().item()
        print("test acc on clean examples (%): {:.3f}".format(correct / nb_test * 100.0))
        print("test acc on FGM adversarial examples (%): {:.3f}".format(correct_fgm / nb_test * 100.0))
        print("test acc on PGD adversarial examples (%): {:.3f}".format(correct_pgd / nb_test * 100.0))
        print("====================================")



if __name__ == "__main__":
    main()
