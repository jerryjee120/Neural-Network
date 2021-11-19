from os.path import join, exists
from os import mkdir
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn


class RepeaterDataset(torch.utils.data.Dataset):
    def __init__(self, total_count, length):
        self.data = torch.randint(0, 2, (total_count, length, 1)).type(torch.FloatTensor)
        self.labels = self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Datasets:
    def __init__(self, batch_size, total_count=1000, length=15):
        self.train_loader = torch.utils.data.DataLoader(RepeaterDataset(total_count, length), batch_size=batch_size,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(RepeaterDataset(total_count, length), batch_size=batch_size * 2,
                                                       shuffle=True)


class RNNAutoencoder(nn.Module):
    def __init__(self, hidden_size=256):
        super(RNNAutoencoder, self).__init__()

        self.hidden_size = hidden_size

        self.rnn1 = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.rnn2 = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        """
        Args:
            x: [N,L,1]
        """
        o1 = x.size()
        o2 = self.rnn1(x)
        o3 = torch.ones(size=o1)
        o4 = self.rnn2(o3, o2[1])
        o5 = self.fc(o4[0])
        o6 = torch.sigmoid(o5)
        return o6


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
        length = self.datasets.test_loader.dataset[0][1].size()[0]
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.datasets.test_loader:
                output = self.model(data)
                test_loss += self.loss_fn(output, target).item() * len(data)
                correct += output.round().eq(target.round()).sum().item()

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
        if not exists(self.results_path):
            mkdir(self.results_path)
        torch.save(self.model.state_dict(), join(self.results_path, 'model.pth'))

    def plot(self):
        import matplotlib.pyplot as plt
        self.train_df[["train_loss", "val_loss"]].ffill().plot(title="loss", grid=True, logy=False)
        self.train_df[["accuracy"]].dropna().plot(title="accuracy", grid=True)
        plt.show()


def train():
    torch.manual_seed(0)
    length = 10
    datasets = Datasets(100, total_count=5000, length=length)

    model = RNNAutoencoder(hidden_size=5)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    trainer = Trainer(datasets, model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=1000)

    for i in range(5):
        test_x = torch.randint(0, 2, (1, length, 1)).type(torch.FloatTensor)
        test_y = test_x
        print(f"----- {i} -----")
        print(test_x.flatten().type(torch.IntTensor).tolist())

        predict = model(test_x)
        print(predict.round().flatten().type(torch.IntTensor).tolist())

        print(f"{predict.round().eq(test_y.round()).sum().item() / length:.0%}")

    trainer.plot()


if __name__ == "__main__":
    train()
