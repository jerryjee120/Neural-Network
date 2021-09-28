from os.path import join
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn


class FC1Model(nn.Module):
    def __init__(self):
        super(FC1Model, self).__init__()

        self.fc = nn.Linear(in_features=15, out_features=50)

    def forward(self, x):
        """
        Args:
            x: [N,1,15]
        """
        o1 = self.fc(x)
        o2 = torch.sigmoid(o1)
        o3 = torch.max(o2, dim=-1)[0]
        o4 = o3.reshape(-1)
        return o4


class Conv1Model(nn.Module):
    def __init__(self):
        super(Conv1Model, self).__init__()

        self.conv = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=7)

    def forward(self, x):
        """
        Args:
            x: [N,1,15]
        """
        o1 = self.conv(x)
        o2 = torch.max(o1, dim=-1)[0]
        o3 = torch.max(o2, dim=-1)[0]
        o4 = torch.sigmoid(o3)
        return o4


class FC2Model(nn.Module):
    def __init__(self):
        super(FC2Model, self).__init__()

        self.fc1 = nn.Linear(in_features=15, out_features=50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=50, out_features=1)

    def forward(self, x):
        """
        Args:
            x: [N,1,15]
        """
        o1 = self.fc1(x)
        o2 = self.relu(o1)
        o3 = self.fc2(o2)
        o4 = torch.sigmoid(o3)
        o5 = o4.reshape(-1)
        return o5


class MultiConvModel(nn.Module):
    def __init__(self):
        super(MultiConvModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3)

    def forward(self, x):
        """
        Args:
            x: [N,1,15]
        """
        o1 = self.conv1(x)
        o2 = self.relu(o1)
        o3 = self.maxpool(o2)
        o4 = self.conv2(o3)
        o5 = torch.max(o4, dim=-1)[0]
        o6 = torch.max(o5, dim=-1)[0]
        o7 = torch.sigmoid(o6)
        return o7


class ObjFunc:
    feature_length = 7
    threshold = 0.3

    def _is_edge(self, xs, start, edge1=True):
        diff = xs[start] - xs[start + 1]
        ret = diff > self.threshold if edge1 else diff < -self.threshold
        return ret

    def _is_feature(self, xs):
        assert len(xs) == self.feature_length

        return self.is_edge(xs, 0) \
            and self._is_edge(xs, 2, edge1=False) \
            and self._is_edge(xs, 3) \
            and self._is_edge(xs, 5, edge1=False)

    def __call__(self, xs):
        """10
        Args:
            xs: a list of double
        return: bool
        """
        assert len(xs) >= self.feature_length

        for i in range(len(xs) - self.feature_length):
            sub_xs = xs[i:i + 7]
            if self._is_feature(sub_xs):
                return True

        return False


class Pattern1DDataset(torch.utils.data.Dataset):
    def __init__(self, total_count, length):
        pos_count = total_count // 2
        neg_count = total_count - pos_count
        pos_samples = []
        neg_samples = []

        obj_func = ObjFunc()
        while len(pos_samples) < pos_count:
            xs = torch.rand(length).tolist()
            y = obj_func(xs)
            if y:
                pos_samples.append(xs)
            elif len(neg_samples) < neg_count:
                neg_samples.append(xs)

        self.data = torch.FloatTensor([pos_samples + neg_samples]).reshape((total_count, 1, length))
        self.labels = torch.FloatTensor([1.0] * pos_count + [0.0] * neg_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Datasets:
    def __init__(self, batch_size, total_count=1000, length=15):
        self.train_loader = torch.utils.data.DataLoader(Pattern1DDataset(total_count, length), batch_size=batch_size,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(Pattern1DDataset(total_count, length), batch_size=batch_size * 2,
                                                       shuffle=True)


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
                correct += output.gt(0.5).eq(target.gt(0.5)).sum().item()

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
        self.train_df[["train_loss", "val_loss"]].ffill().plot(title="loss", grid=True, logy=False)
        self.train_df[["accuracy"]].dropna().plot(title="accuracy", grid=True)
        plt.show()


def train():
    torch.manual_seed(0)
    datasets = Datasets(100, total_count=10000)

    # model = FC1Model()
    # model = Conv1Model()
    # model = FC2Model()
    model = MultiConvModel()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(datasets, model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=100)
    print(list(model.parameters()))
    trainer.plot()


if __name__ == "__main__":
    train()
