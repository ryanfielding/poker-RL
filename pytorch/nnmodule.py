from torch import nn
import torch.nn.functional as F
class net(nn.Module):
    def __init__(self):
        super().__init__()
        #inputs to hidden layer
        self.hidden = nn.Linear(784,256)
        #output layer
        self.output = nn.Linear(256,10)

        #sigmoid
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.hidden(x)
        # x = self.sigmoid(x)
        # x = self.output(x)
        # x = self.softmax(x)
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x))
        return x

#model = net()
