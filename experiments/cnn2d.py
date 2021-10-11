from os.path import join
from tqdm import tqdm
import pandas as pd

import torchvision
import torch
import torch.nn as nn


class Datasets:
    def __init__(self, dataset_path, batch_size):
        self.train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST(dataset_path, train=True, download=True,
                                     transform=torchvision.transforms.ToTensor()),
          batch_size=batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST(dataset_path, train=False, download=True,
                                     transform=torchvision.transforms.ToTensor()),
          batch_size=batch_size * 2, shuffle=True)


class FC1Model(nn.Module):
    def __init__(self):
        super(FC1Model, self).__init__()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=784, out_features=10)

    def forward(self, x):
        """
        Args:
            x: [N,C_{in},H_{in},W_{in}]
        """
        o1 = self.flatten(x)
        o2 = self.fc(o1)
        o3 = torch.log_softmax(o2, dim=-1)
        return o3


class FC2Model(nn.Module):
    def __init__(self):
        super(FC2Model, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=784, out_features=100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        """
        Args:
            x: [N,1,28,28]
        """
        o1 = self.flatten(x)
        o2 = self.fc1(o1)
        o3 = self.relu(o2)
        o4 = self.fc2(o3)
        o5 = torch.log_softmax(o4, dim=-1)
        return o5


class FC3Model(nn.Module):
    def __init__(self):
        super(FC3Model, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=784, out_features=100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        """
        Args:
            x: [N,1,28,28]
        """
        o1 = self.flatten(x)
        o2 = self.fc1(o1)
        o3 = self.relu1(o2)
        o4 = self.fc2(o3)
        o5 = self.relu2(o4)
        o6 = self.fc3(o5)
        o7 = torch.log_softmax(o6, dim=-1)
        return o7


class Conv1Model(nn.Module):
    def __init__(self):
        super(Conv1Model, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=2880, out_features=10)

    def forward(self, x):
        """
        Args:
            x: [N,1,28,28]
        """
        o1 = self.conv(x)
        o2 = self.maxpool(o1)
        o3 = self.relu(o2)
        o4 = self.flatten(o3)
        o5 = self.fc(o4)
        o6 = torch.log_softmax(o5, dim=-1)
        return o6


class Conv2Model(nn.Module):
    def __init__(self):
        super(Conv2Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=640, out_features=10)

    def forward(self, x):
        """
        Args:
            x: [N,1,28,28]
        """
        o1 = self.conv1(x)
        o2 = self.maxpool1(o1)
        o3 = self.relu1(o2)
        o4 = self.conv2(o3)
        o5 = self.maxpool2(o4)
        o6 = self.relu2(o5)
        o7 = self.flatten(o6)
        o8 = self.fc(o7)
        o9 = torch.log_softmax(o8, dim=-1)
        return o9


class Conv3Model(nn.Module):
    def __init__(self):
        super(Conv3Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=4)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=80, out_features=10)

    def forward(self, x):
        """
        Args:
            x: [N,1,28,28]
        """
        o1 = self.conv1(x)
        o2 = self.maxpool1(o1)
        o3 = self.relu1(o2)
        o4 = self.conv2(o3)
        o5 = self.maxpool2(o4)
        o6 = self.relu2(o5)
        o7 = self.conv3(o6)
        o8 = self.relu3(o7)
        o9 = self.flatten(o8)
        o10 = self.fc(o9)
        o11 = torch.log_softmax(o10, dim=-1)
        return o11


class Trainer:
    def __init__(self, datasets, model, optimizer, loss_fn, results_path='results'):
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.results_path = results_path

        self.train_df = None

    def train_epoch(self, msg_format):
        self.model.train()

        losses = []
        bar = tqdm(self.datasets.train_loader)
        for data, target in bar:
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.loss_fn(output, target)

            loss.backward()
            self.optimizer.step()

            bar.set_description(msg_format.format(loss.item()))

            losses.append(loss.item())
        return losses

    def test(self):
        self.model.eval()

        count = len(self.datasets.test_loader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.datasets.test_loader:
                output = self.model(data)
                test_loss += self.loss_fn(output, target).item() * len(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()

        return test_loss / count, correct / count

    def train(self, num_epoch):
        val_loss, accuracy = self.test()
        all_losses = [[None, val_loss, accuracy]]

        for epoch in range(num_epoch):
            # train
            train_losses = self.train_epoch(
                f'train {epoch}/{num_epoch} -- loss: {{:3.2f}}, val_loss: {val_loss:3.2f}, accuracy: {accuracy:.1%}')

            # test
            val_loss, accuracy = self.test()
            all_losses.extend([
                [train_loss, None, None]
                for train_loss in train_losses
            ])
            all_losses.append([None, val_loss, accuracy])

        self.save_model()
        self.train_df = pd.DataFrame(data=all_losses, columns=["train_loss", "val_loss", "accuracy"])
        self.train_df.to_csv(join(self.results_path, "train.csv"), index=False)

    def save_model(self):
        torch.save(self.model.state_dict(), join(self.results_path, 'model.pth'))

    def plot(self):
        import matplotlib.pyplot as plt
        self.train_df[["train_loss", "val_loss"]].ffill().plot(grid=True, logy=True)
        self.train_df[["accuracy"]].dropna().plot(grid=True)
        plt.show()


def train():
    torch.manual_seed(0)

    # model = FC1Model()
    # model = FC2Model()
    # model = FC3Model()
    # model = Conv1Model()
    # model = Conv2Model()
    model = Conv3Model()

    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    trainer = Trainer(Datasets("datasets", 100), model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=15)
    trainer.plot()


if __name__ == "__main__":
    train()
