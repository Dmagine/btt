import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class Monitor():
    def __init__(self):
        self.default_module_type_list = [nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.RNN, nn.LSTM, nn.GRU]
        self.default_metric_suffix_list = ["mid", "rate0"]
        self.hook_handle_list = []
        self.backward_hook_counter = 0
        pass

    def register_hook(self, model):

        def forward_hook_get_feature_value(module: nn.Module, feature_value_in, feature_value_out):
            print(type(module), module.weight.shape if module.weight is not None else None, module.training)

            # feature_value_in tuple len=1
            feature_value_in = feature_value_in[0] if feature_value_in is not None else None

            var_shape = feature_value_in.shape if feature_value_in is not None else -1
            var_first = feature_value_in[0][0:2] if feature_value_in[0] is not None else -1
            print(type(feature_value_in), len(feature_value_in), var_shape, var_first)

            var_shape = feature_value_out.shape if feature_value_out is not None else -1
            var_first = feature_value_out[0][0:2] if feature_value_out[0] is not None else -1
            print(type(feature_value_out), len(feature_value_in), var_shape, var_first)

            print()

        def backward_hook_get_feature_gradient(module: nn.Module, feature_grad_in, feature_grad_out):
            self.backward_hook_counter += 1
            print(self.backward_hook_counter)
            print(type(module), module.weight.shape if module.weight is not None else None)

            var_shape = module.weight.grad.shape if module.weight.grad is not None else -1
            var_first = module.weight.grad[0][0:2] if module.weight.grad is not None else -1
            print(var_shape, var_first)

            var_shape = feature_grad_in[0].shape if feature_grad_in is not None else -1
            print(type(feature_grad_in), feature_grad_in[0].shape, feature_grad_in[1].shape, feature_grad_in[2].shape)
            # exit()
            # var_first = grad_in[0][0][0:2] if grad_in[0] is not None else -1
            print(var_shape)

            var_shape = feature_grad_out[0].shape if feature_grad_out[0] is not None else -1
            # var_first = grad_out[0][0][0:2] if grad_out[0] is not None else -1
            print(var_shape)
            print()

        module_iter = ModuleIterator(model, self.default_module_type_list)
        for module in module_iter:
            h = module.register_forward_hook(forward_hook_get_feature_value)
            self.hook_handle_list.append(h)

            # h = module.register_backward_hook(backward_hook_get_feature_gradient)
            # h = module.register_full_backward_hook(backward_hook_get_weight_gradient)
            self.hook_handle_list.append(h)

    def unregister_hook(self):
        for h in self.hook_handle_list:
            h.remove()


monitor = Monitor()


class ModuleIterator:
    def __init__(self, model, module_type_list):
        self.model = model
        self.default_module_type_list = [nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.RNN, nn.LSTM, nn.GRU] \
            if module_type_list is None else module_type_list

    def __iter__(self):
        # 作为iter时的init方法
        self.module_iter = iter(self.model.modules())
        return self

    def __next__(self):
        while True:
            try:
                tmp_module = next(self.module_iter)
            except StopIteration:
                raise StopIteration
            print(type(tmp_module))
            if type(tmp_module) in self.default_module_type_list:
                return tmp_module


def train(dataloader, model, loss_fn, optimizer):
    monitor.register_hook(model)
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
        monitor.unregister_hook()
        exit()

    print("Done!")


if __name__ == "__main__":
    main()
