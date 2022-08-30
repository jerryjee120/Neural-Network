import math
from random import seed, random


# the objective function
def o(x, y):
    return 1.0 if x*x + y*y < 1 else 0.0


# samples
sample_density = 10
xs = [
    [-2.0 + 4 * x/sample_density, -2.0 + 4 * y/sample_density]
    for x in range(sample_density+1)
    for y in range(sample_density+1)
]
dataset = [
    (x, y, o(x, y))
    for x, y in xs
]


# activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    _output = sigmoid(x)
    return _output * (1 - _output)


seed(0)


# neural network
class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random()-0.5 for _ in range(num_inputs)]
        self.bias = 0.0

        # caches
        self.z_cache = None
        self.inputs_cache = None

    def forward(self, inputs):
        assert len(inputs) == len(inputs)
        self.inputs_cache = inputs

        # z = wx + b
        self.z_cache = sum([
            i * w
            for i, w in zip(inputs, self.weights)
        ]) + self.bias
        # sigmoid(wx + b)
        return sigmoid(self.z_cache)

    def zero_grad(self):
        self.d_weights = [0.0 for w in self.weights]
        self.d_bias = 0.0

    def backward(self, d_a):
        d_loss_z = d_a * sigmoid_derivative(self.z_cache)
        self.d_bias += d_loss_z
        for i in range(len(self.inputs_cache)):
            self.d_weights[i] += d_loss_z * self.inputs_cache[i]
        return [d_loss_z * w for w in self.weights]

    def update_params(self, learning_rate):
        self.bias -= learning_rate * self.d_bias
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.d_weights[i]

    def params(self):
        return self.weights + [self.bias]


class MyNet:
    def __init__(self, num_inputs, hidden_shapes):
        layer_shapes = hidden_shapes + [1]
        input_shapes = [num_inputs] + hidden_shapes
        self.layers = [
            [
                Neuron(pre_layer_size)
                for _ in range(layer_size)
            ]
            for layer_size, pre_layer_size in zip(layer_shapes, input_shapes)
        ]

    def forward(self, inputs):
        for layer in self.layers:
            inputs = [
                neuron.forward(inputs)
                for neuron in layer
            ]
        # return the output of the last neuron
        return inputs[0]

    def zero_grad(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.zero_grad()

    def backward(self, d_loss):
        d_as = [d_loss]
        for layer in reversed(self.layers):
            da_list = [
                neuron.backward(d_a)
                for neuron, d_a in zip(layer, d_as)
            ]
            d_as = [sum(da) for da in zip(*da_list)]

    def update_params(self, learning_rate):
        for layer in self.layers:
            for neuron in layer:
                neuron.update_params(learning_rate)

    def params(self):
        return [[neuron.params() for neuron in layer]
                for layer in self.layers]


# loss function
def square_loss(predict, target):
    return (predict-target)**2


def square_loss_derivative(predict, target):
    return 2 * (predict-target)


# build neural network
net = MyNet(2, [4])
print(net.forward([0, 0]))
targets = [z for x, y, z in dataset]


# train
def one_step(learning_rate):
    net.zero_grad()

    loss = 0.0
    num_samples = len(dataset)
    for x, y, z in dataset:
        predict = net.forward([x, y])
        loss += square_loss(predict, z)

        net.backward(square_loss_derivative(predict, z) / num_samples)

    net.update_params(learning_rate)
    return loss / num_samples


def train(epoch, learning_rate):
    for i in range(epoch):
        loss = one_step(learning_rate)
        if i == 0 or (i + 1) % 100 == 0:
            print(f"{i + 1} {loss:.4f}")


def inference(x, y):
    return net.forward([x, y])


train(2000, learning_rate=10)
inference(1, 2)
