import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Module
from options import args_parser

args = args_parser()

class MLP_adult(Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_adult, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 第一层，全连接
        self.dropout = nn.Dropout()  # dropout是为了防止过拟合而设置的, Dropout只能用在训练部分而不能用在测试部分，Dropout一般用在全连接网络映射层之后
        self.relu = nn.ReLU()#relu的激活函数，F.relu是函数调用，一般使用在foreward函数中，而nn.ReLU()是模块调用，一般在定义网络层的时候使用
        self.fc2 = nn.Linear(64, 32)         # 第二层
        self.fc3 = nn.Linear(32, output_dim) # 输出层
        self.softmax = nn.Softmax(dim=1)#对n维输入张量运行softmax函数，将张量的每个元素缩放到(0,1)区间且和为1.

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
    

class MLP(Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        #super().__int__()
        self.layer_input = nn.Linear(dim_in, 50)
        self.dropout = nn.Dropout()  # dropout是为了防止过拟合而设置的, Dropout只能用在训练部分而不能用在测试部分，Dropout一般用在全连接网络映射层之后
        self.relu = nn.ReLU()#relu的激活函数，F.relu是函数调用，一般使用在foreward函数中，而nn.ReLU()是模块调用，一般在定义网络层的时候使用
        self.layer_hidden = nn.Linear(50, dim_out)
        self.softmax = nn.Softmax(dim=1)#对n维输入张量运行softmax函数，将张量的每个元素缩放到(0,1)区间且和为1.

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x
        #self.softmax(x)



class MLPCifar(Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLPCifar, self).__init__()
        #super().__int__()
        self.layer_input = nn.Linear(dim_in, 50)
        self.dropout = nn.Dropout()  # dropout是为了防止过拟合而设置的, Dropout只能用在训练部分而不能用在测试部分，Dropout一般用在全连接网络映射层之后
        self.relu = nn.ReLU()#relu的激活函数，F.relu是函数调用，一般使用在foreward函数中，而nn.ReLU()是模块调用，一般在定义网络层的时候使用
        self.layer_hidden = nn.Linear(50, dim_out)
        self.softmax = nn.Softmax(dim=1)#对n维输入张量运行softmax函数，将张量的每个元素缩放到(0,1)区间且和为1.

    def forward(self, x):
        x = x.reshape(-1, x.shape[-3]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x




class CNNMnist(nn.Module):
    def __init__(self,dim_out):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)
        #二维卷积方法，相对应的还有一维卷积方法nn.Conv1d,常用于文本数据的处理，而nn.Conv2d一般用于二维图像
        #in_channel表示的输入图像的通道数，out_channel表示卷积产生的通道数，kernel_size表示的卷积核大小
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.conv2_drop = nn.Dropout2d()
        #dropout是对每一个数都来一次随机置为0，把输入看成由一个个数字组成，dropout2d：将输入看成一个个矩阵组成，其应用场景通常是图片，对于每个通道置为0
        self.fc1 = nn.Linear(int(1344), 50)
        #self.fc1 = nn.Linear(int(2048), 50)
        self.fc2 = nn.Linear(50, dim_out)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #激活层
        #pool层用于提取重要信息的操作，可以去掉部分相邻的信息，减少计算开销
        #MaxPool在提取数据时，保留相邻信息中的最大值，去掉其他值
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 1))
       # print(x.shape)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        #这里的training表示的是个二进制，表示训练或者测试
        x = self.fc2(x)
        return x

  


class CNNcifar(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CNNcifar, self).__init__()
        #初始化nn.Module类中的属性
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1)
        #二维卷积方法，相对应的还有一维卷积方法nn.Conv1d,常用于文本数据的处理，而nn.Conv2d一般用于二维图像
        #in_channel表示的输入图像的通道数，out_channel表示卷积产生的通道数，kernel_size表示的卷积核大小
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.conv2_drop = nn.Dropout2d()
        #dropout是对每一个数都来一次随机置为0，把输入看成由一个个数字组成，dropout2d：将输入看成一个个矩阵组成，其应用场景通常是图片，对于每个通道置为0
        self.fc1 = nn.Linear(int(128*dim_in*2), 50)
        #self.fc1 = nn.Linear(int(2048), 50)
        # self.fc2 = nn.Linear(50, 10)
        self.fc2 = nn.Linear(50, dim_out)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #激活层
        #pool层用于提取重要信息的操作，可以去掉部分相邻的信息，减少计算开销
        #MaxPool在提取数据时，保留相邻信息中的最大值，去掉其他值
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 1))
       # print(x.shape)
        x = x.view(-1, x.shape[-3]*x.shape[-2]*x.shape[-1])
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        #这里的training表示的是个二进制，表示训练或者测试
        x = self.fc2(x)
        return x



class Lenetmnist(nn.Module):
    def __init__(self, dim_out):
        super(Lenetmnist, self).__init__()
        # self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc1 = nn.Linear(4608, 50)
        self.fc2 = nn.Linear(50, 192)
        self.fc3 = nn.Linear(192, dim_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x1))
        x = self.fc3(x)
        return x




class Lenetcifar(nn.Module):
    def __init__(self, dim_out):
        super(Lenetcifar, self).__init__()
        self.n_cls = 100
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        # self.flatten_size = self._get_flatten_size()
        self.fc1 = nn.Linear(50176, 50)
        self.fc2 = nn.Linear(50, 192)
        self.fc3 = nn.Linear(192, dim_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[-3] * x.shape[-2] * x.shape[-1])
        x1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x1))
        x = self.fc3(x)
        return x





if __name__ == "__main__":
    # dim_in = 28
    # dim_hidden = 64
    # dim_out =10
    # #models = MLP(dim_in, dim_hidden, dim_out)
    # models = Lenetmnist()
    # print(models)
    #print(x.type())
    models = MLP(12,23,12)
    print(models)
    # x = torch.randn(64, 1, 2, 2)  # 个数、通道数、宽、高
    # print(x)
    # y = models(x)
    # y1 = models.forward2(y,True)
    # print(y.size())