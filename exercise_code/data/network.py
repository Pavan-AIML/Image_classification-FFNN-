import torch.nn as nn

class Net(nn.Module):
    def __init__(self, activation=nn.Sigmoid(), input_size=1 * 28 * 28, hidden_size=100, classes=10):
        super(Net, self).__init__()
        self.input_size = input_size

        # Here we initialize our activation and set up our two linear layers
        self.activation = activation
        self.softmax = nn.Softmax()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)  # flatten for the input neurans in the input layer.
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
