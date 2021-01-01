import matplotlib.pyplot as plt
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
    should have a size of 50, both should be followed by ReLU activation function.
    """

    def __init__(self, image_size):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.name = "ModelA"

    def forward(self, x):
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
        super(ModelB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
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

    def __init__(self, image_size, dropout):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.do0 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(100, 50)
        self.do1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
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
        super(ModelD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.bn0 = nn.BatchNorm1d(num_features=100)
        self.fc1 = nn.Linear(100, 50)
        self.bn1 = nn.BatchNorm1d(num_features=50)
        self.fc2 = nn.Linear(50, 10)
        self.bn2 = nn.BatchNorm1d(num_features=10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return F.log_softmax(x, dim=1)


class ModelE(nn.Module):
    """
    Model E - Neural Network with ﬁve hidden layers:[128,64,10,10,10] using ReLU .
    """

    def __init__(self, image_size):
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
    class that manage the network
    """

    def __init__(self, model, image_size=28 * 28, optimizer=optim.SGD, dropout=None, learning_rate=0.40, batch_size=64):
        """
        init the object
        :param model: which model to run a,b,c,d,e or f
        :param image_size:
        :param optimizer: which oprimaizer to run
        :param dropout: if it use dropout, the dropout rate
        :param epoch: number of epochs
        :param learning_rate:
        :param batch_size:
        """
        self.image_size = image_size
        self.epoch = 10
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.print_debug = False
        self.total_loss = []
        self.total_accuracy = []
        if dropout:
            self.Net = model(image_size, dropout=dropout)
        else:
            self.Net = model(image_size)
        self.optimizer = optimizer(self.Net.parameters(), lr=learning_rate)

    def step_train(self, train_loader, epoch_i=10):
        """
        train the model
        :param train_loader: dataset
        :param epoch_i: in which epoch am i
        :return:
        """
        total_loss = 0
        self.Net.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            self.optimizer.zero_grad()
            output = self.Net.forward(data)
            loss = F.nll_loss(output, labels)
            loss.backward()
            self.optimizer.step()
            self.until_now = batch_idx * self.batch_size
            self.data_set_len = len(train_loader.dataset)
            total_loss += loss

    def validate(self, test_loader):
        """
        test the modal
        :param test_loader:
        :return: the accuracy of this epoch
        """
        self.Net.eval()
        test_loss = 0
        correct = 0
        total_loss = 0
        total_curr = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.Net(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).cpu().sum()
            test_loss /= len(test_loader.dataset)
            total_loss += test_loss
            curr_accuracy = 100. * correct / len(test_loader.dataset)
            total_curr += curr_accuracy
            self.total_loss.append(test_loss)
            self.total_accuracy.append(curr_accuracy)

    def train_and_vaildate(self, train_loader, test_loader):
        """
        train and validate in each epoch
        :param train_loader:
        :param test_loader:
        :return:
        """
        for epoch_i in range(self.epoch):
            self.step_train(train_loader, epoch_i)
            self.validate(test_loader)

    def do_train(self, train_loader):
        """
        train in each epoch
        :param train_loader:
        :param test_loader:
        :return:
        """
        for epoch_i in range(self.epoch):
            self.step_train(train_loader, epoch_i)

    def test(self, test_input):
        """
        test the modal
        :param test_input:
        :return: the accuracy of this epoch
        """
        self.Net.eval()
        with torch.no_grad():
            return self.Net(test_input).argmax()


def load_data():
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
    train_loader, val_loader, trans, test_loader = load_data()
    test_x_data = np.loadtxt("test_x") / 255
    test_x_data = trans(test_x_data).float()

    netA = NeuralNetwork(model=ModelA, optimizer=optim.SGD, learning_rate=0.325)
    netB = NeuralNetwork(model=ModelB, optimizer=optim.Adadelta, learning_rate=0.038)
    netC = NeuralNetwork(model=ModelC, optimizer=optim.SGD, dropout=1 / 2, learning_rate=0.1)
    netD = NeuralNetwork(model=ModelD, optimizer=optim.SGD, learning_rate=0.01)
    netE = NeuralNetwork(model=ModelE, optimizer=optim.SGD, learning_rate=0.01)
    netF = NeuralNetwork(model=ModelF, optimizer=optim.Adam, learning_rate=0.0094)

    netD.print_debug = False
    netD.do_train(train_loader)
    with open("test_y", "w") as file:
        for test_input in test_x_data[0]:
            predict_class = netD.test(test_input)
            file.write(str(int(predict_class)))
            file.write("\n")


if __name__ == "__main__":
    main()
