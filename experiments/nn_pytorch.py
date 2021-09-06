import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# the objective function
def o(x, y):
    return 1.0 if x * x + y * y < 1 else 0.0


# samples
sample_density = 10
xs = torch.FloatTensor([
    [-2.0 + 4 * x / sample_density, -2.0 + 4 * y / sample_density]
    for x in range(sample_density + 1)
    for y in range(sample_density + 1)
])
ys = torch.FloatTensor([
    [o(x, y)]
    for x, y in xs
])


# model
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.fc1 = nn.Linear(in_features=2, out_features=4)
        self.fc2 = nn.Linear(in_features=4, out_features=1)

    def forward(self, x):
        o2 = self.fc1(x)
        o3 = torch.sigmoid(o2)
        o5 = self.fc2(o3)
        o4 = torch.sigmoid(o5)
        return o4


torch.manual_seed(0)

# build neural network
net = MyNet()
x = torch.FloatTensor([[0, 0]])
print(net.forward(x))


# train
def one_step(optimizer):
    optimizer.zero_grad()

    output = net(xs)
    loss = F.mse_loss(output, ys)

    loss.backward()
    optimizer.step()

    return loss


def train(epoch, learning_rate):
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    for i in range(epoch):
        loss = one_step(optimizer)
        if i == 0 or (i+1) % 100 == 0:
            print(f"{i+1} {loss:.4f}")


train(2000, learning_rate=10)
