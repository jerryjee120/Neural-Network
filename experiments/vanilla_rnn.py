from os.path import join, exists
from os import mkdir
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn


def obj_func(xs):
    """
    Args:
        xs: [L], a list of int
    return:
        [L], a list of int
    """
    results = []
    _c = 0
    for x in xs:
        _c += x.item()
        results.append(_c)
    return results


class CounterDataset(torch.utils.data.Dataset):
    def __init__(self, total_count, length):
        self.data = torch.randint(0, 2, (total_count, length, 1)).type(torch.FloatTensor)
        self.labels = torch.unsqueeze(torch.FloatTensor([
            obj_func(xs.flatten())
            for xs in self.data
        ]), -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Datasets:
    def __init__(self, batch_size, total_count=1000, length=15):
        self.train_loader = torch.utils.data.DataLoader(CounterDataset(total_count, length), batch_size=batch_size,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(CounterDataset(total_count, length), batch_size=batch_size * 2,
                                                       shuffle=True)


class VanillaRNN(nn.Module):
    def __init__(self, hidden_size=1):
        super(VanillaRNN, self).__init__()

        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, nonlinearity='relu', batch_first=True)

    def forward(self, x):
        """
        Args:
            x: [N,10,1]
        """
        o1 = self.rnn(x)
        return o1[0]


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
    torch.manual_seed(3)
    datasets = Datasets(100, total_count=1000)

    model = VanillaRNN()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(datasets, model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=40)
    params = list(model.parameters())
    print(params)
    w1 = params[0][0][0].item()
    w2 = params[1][0][0].item()
    b1 = params[2][0].item()
    b2 = params[3][0].item()
    print(f"relu({w1:.3f}x{b1:+.3f}{w2:+.3f}h{b2:+.3f})")

    trainer.plot()


if __name__ == "__main__":
    train()
