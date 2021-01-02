import sys
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision import datasets
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ModelA(nn.Module):
    """
    Model A - Neural Network with two hidden layers,
    the first layer should have a size of 100 and the second layer
    should have a size of 50, both should be followed by ReLU activation
    function.
    """
    def __init__(self, image_size):
        """
        init model A
        """
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        forward propagation function of model A - using ReLU activation
        function
        """
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelB(nn.Module):
    """
    Model B - Neural Network with two hidden layers, the first layer should
    have a size of 100 and the second layer should have a size of 50, both
    should be followed by ReLU activation function, train this model
    with ADAM optimizer.
    """
    def __init__(self, image_size):
        """
        init model B
        """
        super(ModelB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        forward propagation function of model B - using ReLU activation
        function
        """
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelC(nn.Module):
    """
    Model C - Dropout – add dropout layers to model A.
    You should place the dropout on the output of the hidden layers
    """
    def __init__(self, image_size, dropout_rate):
        """
        init model C
        """
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.do0 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(100, 50)
        self.do1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        forward propagation function of model C - using ReLU activation
        function and dropout
        """
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.do0(x)
        x = F.relu(self.fc1(x))
        x = self.do1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelD(nn.Module):
    """
     Model D - Batch Normalization - add Batch Normalization layers to model A.
     You should place the Batch Normalization before the activation functions
    """
    def __init__(self, image_size):
        """
        init model D
        """
        super(ModelD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.bn0 = nn.BatchNorm1d(num_features=100)
        self.fc1 = nn.Linear(100, 50)
        self.bn1 = nn.BatchNorm1d(num_features=50)
        self.fc2 = nn.Linear(50, 10)
        self.bn2 = nn.BatchNorm1d(num_features=10)

    def forward(self, x):
        """
        forward propagation function of model D - using batch normalization and
        ReLU activation function
        """
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return F.log_softmax(x, dim=1)


class ModelE(nn.Module):
    """
    Model E - Neural Network with ﬁve hidden layers:[128,64,10,10,10] using
    ReLU.
    """
    def __init__(self, image_size):
        """
        init model E
        """
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)

    def forward(self, x):
        """
        forward propagation function of model E - using ReLU activation
        function
        """
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)


class ModelF(nn.Module):
    """
    Model F - Neural Network with five hidden layers:[128,64,10,10,10] using
    Sigmoid.
    """
    def __init__(self, image_size):
        """
        init model F
        """
        super(ModelF, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)

    def forward(self, x):
        """
        forward propagation function of model F - using Sigmoid activation
        function
        """
        x = x.view(-1, self.image_size)
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)


class NeuralNetwork:
    """
    A class that represents the neural network
    """
    def __init__(self, model, image_size=28 * 28, optimizer=optim.SGD,
                 dropout_rate=None, learning_rate=0.40, batch_size=64):
        """
        Initialize the network params and model
        :param model: model to run a,b,c,d,e,f
        :param image_size: the image size, default 28*28
        :param optimizer: which optimizer to run
        :param dropout_rate: the dropout rate used
        :param learning_rate: the learning rate used
        :param batch_size: the batch size used
        """
        self.image_size = image_size
        self.epochs = 10
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.print_debug = False
        self.total_loss = []
        self.total_accuracy = []
        if dropout_rate:
            self.Network = model(image_size, dropout=dropout_rate)
        else:
            self.Network = model(image_size)
        self.optimizer = optimizer(self.Network.parameters(), lr=learning_rate)

    def train_step(self, train_loader):
        """
        train the model
        :param train_loader: dataset
        :return:
        """
        total_loss = 0
        self.Network.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            self.optimizer.zero_grad()
            output = self.Network.forward(data)
            loss = F.nll_loss(output, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss

    def validate(self, test_loader):
        """
        validate the model and return the accuracy of this epoch
        """
        self.Network.eval()
        test_loss = 0
        correct = 0
        total_loss = 0
        total_curr = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.Network(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, size_average=False).item()
                # gets the index of the max log-probability
                prediction = output.max(1, keepdim=True)[1]
                correct += prediction.eq(target.view_as(prediction)).cpu().sum()
            test_loss /= len(test_loader.dataset)
            total_loss += test_loss
            accuracy = 100. * correct / len(test_loader.dataset)
            total_curr += accuracy
            self.total_loss.append(test_loss)
            self.total_accuracy.append(accuracy)

    def train(self, train_loader):
        """
        train in each epoch
        """
        for num_epoch in range(self.epochs):
            self.train_step(train_loader)

    def test(self, test_input):
        """
        test the model and returns the result of this epoch
        """
        self.Network.eval()
        with torch.no_grad():
            return self.Network(test_input).argmax()


def load_data():
    """
    loads the fashionMNIST dataset
    """
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

    fashion = datasets.FashionMNIST("./data", train=True, download=True, transform=trans)
    train_set, val_set = torch.utils.data.random_split(fashion, [round(len(fashion) * 0.8),
                                                                 len(fashion) - round(len(fashion) * 0.8)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)

    return train_loader, val_loader, trans, test_loader


def main():
    # get parameters
    train_x_file = sys.argv[1]
    train_y_file = sys.argv[2]
    test_x_file = sys.argv[3]# should be "test_x"

    train_loader, val_loader, trans, test_loader = load_data()
    test_x_data = np.loadtxt(test_x_file) / 255
    test_x_data = trans(test_x_data).float()

    #netA = NeuralNetwork(model=ModelA, optimizer=optim.SGD, learning_rate=0.325)
    #netB = NeuralNetwork(model=ModelB, optimizer=optim.Adadelta, learning_rate=0.038)
    #netC = NeuralNetwork(model=ModelC, optimizer=optim.SGD, dropout=1 / 2, learning_rate=0.1)
    netD = NeuralNetwork(model=ModelD, optimizer=optim.SGD, learning_rate=0.01)
    #netE = NeuralNetwork(model=ModelE, optimizer=optim.SGD, learning_rate=0.01)
    #netF = NeuralNetwork(model=ModelF, optimizer=optim.Adam, learning_rate=0.0094)

    netD.print_debug = False
    netD.train(train_loader) # train the best model
    # run test and write the results to the file
    with open("test_y", "w") as file:
        for test_input in test_x_data[0]:
            predict_class = netD.test(test_input)
            file.write(str(int(predict_class)))
            file.write("\n")


if __name__ == "__main__":
    main()