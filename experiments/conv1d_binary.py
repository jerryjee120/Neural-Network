from os.path import join
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn


class Net1d(nn.Module):
    def __init__(self):
        super(Net1d, self).__init__()

        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2)

    def forward(self, x):
        """
        Args:
            x: [N,1,100]
        """
        o1 = self.conv(x)
        o4 = torch.sigmoid(o1)
        return o4


def obj_func(in_tensor):
    """1->0
    Args:
        in_tensor: [N,1,100]
    return:
        [N, 1, C]
    """
    return torch.FloatTensor([
        [
            [
                1 if seq[i + 1] < x else 0
                for i, x in enumerate(seq[:-1])
            ]
            for seq in xs
        ]
        for xs in in_tensor
    ])


class PairUpDataset(torch.utils.data.Dataset):
    def __init__(self, total_count, length):
        self.data = torch.randint(0, 2, (total_count,1,length)).type(torch.FloatTensor)
        self.labels = obj_func(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Datasets:
    def __init__(self, batch_size, total_count=1000, length=10):
        self.length = length
        self.train_loader = torch.utils.data.DataLoader(
          PairUpDataset(total_count, length),
          batch_size=batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
          PairUpDataset(total_count, length),
          batch_size=batch_size * 2, shuffle=True)


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
        length = self.datasets.test_loader.dataset[0][1].size()[-1]
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.datasets.test_loader:
                output = self.model(data).gt(0.5)
                test_loss += self.loss_fn(output, target).item() * len(data)
                correct += output.eq(target).sum().item()

        return test_loss / count, correct / count / length

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

    model = Net1d()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    trainer = Trainer(Datasets(100), model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=10)
    print(list(model.parameters()))
    trainer.plot()


if __name__ == "__main__":
    train()
