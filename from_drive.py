import torch
from torchvision import transforms
from torchvision import datasets
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FirstNet(nn.Module):
    """
    from the presentaion
    """
    def __init__(self,image_size,fc0_size=128,fc1_size=128,fc2_size=10):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, fc0_size)
        self.fc1 = nn.Linear(fc0_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)
    @staticmethod
    def name():
        return "FirstNet"

class ModelA(nn.Module):
    """
    Model A - Neural Network with two hidden layers,
    the ﬁrst layer should have a size of 100 and the second layer
    should have a size of 50, both should be followed by ReLU activation function.
    """
    def __init__(self,image_size,fc0_size=100,fc1_size=50,fc2_size=10):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, fc0_size)
        self.fc1 = nn.Linear(fc0_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x))

    @staticmethod
    def name():
        return "ModelA"

class ModelB(nn.Module):
    """
    Model B - Dropout – add dropout layers to model A.
    You should place the dropout on the output of the hidden layers
    """

    def __init__(self,image_size,dropout,fc0_size=100,fc1_size=50,fc2_size=10):
        super(ModelB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, fc0_size)
        self.do0 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(fc0_size, fc1_size)
        self.do1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(fc1_size, fc2_size)


    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.do0(x)
        x = F.relu(self.fc1(x))
        x = self.do1(x)
        return F.log_softmax(self.fc2(x))

    @staticmethod
    def name():
        return "ModelB"

class ModelC(nn.Module):

    """
     Model C - Batch Normalization - add Batch Normalization layers to model A.
      You should place the Batch Normalization before the activation functions

    """
    def __init__(self,image_size,fc0_size=100,fc1_size=50,fc2_size=10):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, fc0_size)
        self.bn0 = nn.BatchNorm1d(num_features=fc0_size)
        self.fc1 = nn.Linear(fc0_size, fc1_size)
        self.bn1 = nn.BatchNorm1d(num_features=fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.bn2 = nn.BatchNorm1d(num_features=fc2_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        return F.log_softmax(self.bn2 (self.fc2(x)))

    @staticmethod
    def name():
        return "ModelC"

class ModelD(nn.Module):
    """
    Model D - Neural Network with ﬁve hidden layers:[128,64,10,10,10] using ReLU .
    """
    def __init__(self,image_size):
        super(ModelD, self).__init__()
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

        return F.log_softmax(self.fc6(x))

    @staticmethod
    def name():
        return "ModelD"

class ModelE(nn.Module):
    def __init__(self,image_size):
        #[128,64,10,10,10]
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
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return F.log_softmax(self.fc6(x))

    @staticmethod
    def name():
        return "ModelE"

class myNet:
    """
    class that manage the network
    """
    def __init__(self, model, image_size=28*28,optimizer=optim.SGD,dropout=None, epoch=10, learning_rate=0.40, batch_size=64):
        """
        init the object
        :param model:  which model to run a,b,c,d or e
        :param image_size:
        :param optimizer: which oprimaizer to run
        :param dropout: if it use dropout, the dropout rate
        :param epoch:
        :param learning_rate:
        :param batch_size:
        """
        self.image_size = image_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.print_debug = False
        self.total_loss = []
        self.total_accuracy = []
        # self.Net = FirstNet(image_size, fc0_size, fc1_size, fc2_size)
        if dropout:
            self.Net = model(image_size,dropout=dropout)
        else:
            self.Net = model(image_size)
        self.optimizer = optimizer(self.Net.parameters(), lr=learning_rate)

    def step_train(self, train_loader, epoch_i =10):
        """
        train the model
        :param train_loader: dataset
        :param epoch_i: in which epoch am i
        :return:
        """
        self.Net.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            self.optimizer.zero_grad()
            output = self.Net.forward(data)
            loss = F.nll_loss(output, labels)
            loss.backward()
            self.optimizer.step()
            self.until_now =batch_idx * self.batch_size
            self.data_set_len = len(train_loader.dataset)
            if self.print_debug:
                print(f"\tTrain Epoch: {epoch_i} [{self.until_now}/{self.data_set_len}"
                      f" ({round(100. * self.until_now / self.data_set_len, 2)}%)]  loss: {loss}")


    def validate(self, test_loader):
        """
        test the modal
        :param test_loader:
        :return: the accuracy of this epoch
        """
        self.Net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.Net(data)
                test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).cpu().sum()
            test_loss /= len(test_loader.dataset)
            curr_accuracy = 100. * correct / len(test_loader.dataset)
            self.total_loss.append(test_loss)
            self.total_accuracy.append(curr_accuracy)
            print("\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss, correct, len(test_loader.dataset),curr_accuracy))
            return curr_accuracy

    def train_and_vaildate(self,train_loader, test_loader):
        """
        train and validate in each epoch
        :param train_loader:
        :param test_loader:
        :return:
        """
        for epoch_i in range(self.epoch):
            print(f"[!]Train Epoch: {epoch_i}")
            self.step_train(train_loader, epoch_i)
            self.validate(test_loader)


    def do_train(self,train_loader):
        """
        train in each epoch
        :param train_loader:
        :param test_loader:
        :return:
        """
        for epoch_i in range(self.epoch):
            print(f"[!]Train Epoch: {epoch_i}")
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


    def showGraphs(self):
        """
        show the required graphs
        :return:
        """
        import matplotlib.pyplot as plt
        plt.plot(range(1,self.epoch+1), self.total_loss, label=f'Loss - {self.Net.name()} ')
        plt.legend(bbox_to_anchor=(1.0, 1.00))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.xticks(plt_learning)
        plt.show()

        plt.plot(range(1,self.epoch+1), self.total_accuracy, label=f'Accuracy - {self.Net.name()} ')
        plt.legend(bbox_to_anchor=(1.0, 1.00))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        # plt.xticks(plt_learning)
        plt.show()


def best_Values_for_modelB(train_loader,test_loader):
    """
    find the best args for model ModelB
    :param train_loader:
    :param test_loader:
    :return:
    """
    best_accuracy =0
    best_lr = 0
    best_optimazier = ""
    best_droput = 0
    optimazierz ={"SGD":optim.SGD,"ADAM":optim.Adam,"RMSprop":optim.RMSprop,"AdaDelta":optim.Adadelta}
    for optimazier_name,optimazier_func in optimazierz.items():
        print(f"[!]Checking {optimazier_name}")
        for droput in np.arange(0.05, 1, 0.05):
            for lr in np.arange(0.05,1,0.05):
                net_to_check = myNet(model=ModelB,optimizer=optimazier_func,learning_rate=lr,dropout=droput)
                net_to_check.step_train(train_loader)
                print(f"[+]test on:lr: {lr} optimaize: {optimazier_name} dropout: {droput}")
                curr_accuracy = net_to_check.validate(test_loader)
                if best_accuracy < curr_accuracy:
                    best_accuracy = curr_accuracy
                    best_lr = lr
                    best_optimazier = optimazier_name
                    best_droput = droput
                    print(f"[!!!]found better parrams:  accuracy: {best_accuracy } lr: {lr} optimaize: {best_optimazier} dropout: {droput}")
    print (best_accuracy,best_lr,best_optimazier,best_droput)
    return (best_accuracy,best_lr,best_optimazier,best_droput)


def best_Values_for_model_without_droupout(train_loader, test_loader, model):
    """
    find the best args for the models
    :param train_loader:
    :param test_loader:
    :param model: the model to run
    :return:
    """
    best_accuracy =0
    best_lr = 0
    best_optimazier = ""
    optimazierz ={"SGD":optim.SGD,"ADAM":optim.Adam,"RMSprop":optim.RMSprop,"AdaDelta":optim.Adadelta}
    for optimazier_name,optimazier_func in optimazierz.items():
        print(f"[!]Checking {optimazier_name}")
        # for lr in np.arange(0.05,1,0.05):
        #     if optimazier_name == "RMSprop":
        #         lr /= 100
        for lr in np.arange(0.005, 0.5, 0.005):
            net_to_check = myNet(model=model, optimizer=optimazier_func, learning_rate=lr)
            net_to_check.step_train(train_loader)
            print(f"[+]test on:lr: {lr} optimaize: {optimazier_name}")
            curr_accuracy = net_to_check.validate(test_loader)
            if best_accuracy < curr_accuracy:
                best_accuracy = curr_accuracy
                best_lr = lr
                best_optimazier = optimazier_name
                print(f"[!!!]found better parrams:  accuracy: {best_accuracy } lr: {lr} optimaize: {best_optimazier} ")
    print (best_accuracy,best_lr,best_optimazier)
    return (best_accuracy,best_lr,best_optimazier)
# best_Values_for_modelB(train_loader,test_loader)

def find_values_for_model(train_loader,test_loader,model):
    optimazierz ={"SGD":optim.SGD,"ADAM":optim.Adam,"RMSprop":optim.RMSprop,"AdaDelta":optim.Adadelta}
    print(f"[+]testing model: {model.name()}")

    if model.name() == "ModelB":
        acc, lr, opt ,dropout = best_Values_for_modelB(train_loader, test_loader)
        net = myNet(model, learning_rate=lr, optimizer=optimazierz[opt],dropout=dropout)

    else:
        dropout = 1
        acc,lr,opt = best_Values_for_model_without_droupout(train_loader,test_loader,model)

        net = myNet(model, learning_rate=lr, optimizer=optimazierz[opt])
    net.train_and_vaildate(train_loader, test_loader)
    net.showGraphs()
    print(f"[!!!]best_accuracy:{acc},best_lr:{lr},best_optimazier:{opt} dropout:{dropout} ")




transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])


fashion = datasets.FashionMNIST("./data", train=True, download=True,transform=transforms)
train_set, val_set = torch.utils.data.random_split(fashion, [round(len(fashion)*0.8), len(fashion)-round(len(fashion)*0.8)])
train_loader = torch.utils.data.DataLoader(train_set,batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_set,batch_size=64, shuffle=True)

# train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST("./data", train=True, download=True,
#                                                           transform=transforms),batch_size=64, shuffle=True)
# #
# test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST ("./data", train=False,
#                                                          transform=transforms),batch_size=64, shuffle=True)

# find_values_for_model(train_loader,test_loader,ModelC)

# net = myNet(ModelB, learning_rate=0.85, optimizer=optim.Adadelta, dropout=0.1)
# net.train_and_vaildate(train_loader, test_loader)
# net.showGraphs()

# load the test set
test_x_data=np.loadtxt("test_x") / 255
test_x_data = transforms(test_x_data).float()

net = myNet(model=ModelC, optimizer=optim.Adadelta, learning_rate=0.325)
net.print_debug = False
net.do_train(train_loader)
with open("test_y","w") as file:
    for test_input in test_x_data[0]:
        predict_class = net.test(test_input)
        file.write(str(int(predict_class)))
        file.write("\n")