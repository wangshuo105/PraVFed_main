# 1. 本地模型前向训练
# 2. 聚合本地模型参数
# 3. 将本地模型参数发送给其他客户端
# 4. 聚合最终的结果，并进行损失函数计算
# 5. 将损失函数传输给其他客户端
# 6. 本地模型参数更新
# 7. 联合预算准确率
import copy
import time
import os
import torch.optim
import tqdm as tqdm
from torch import nn
from torch.utils.data import DataLoader

from models.models import MLP, CNNMnist, Lenetmnist, Lenetcifar, CNNcifar, MLPCifar
from options import args_parser
from data.data_process import Data
import pandas as pd
from data.data_process import *
from models.resnet_cifar import *
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_folder(folder_name):
    try:
        os.makedirs(folder_name)
        #print(f"Fold{folder_name} finish")
    except FileExistsError:
        print(f"f folder {folder_name} exist")

class active_cifar(object):
    def __init__(self,logger):
        self.labels = None
        self.model = None
        # self.data_loader = data_loader
        self.global_params = None
        self.optimizer = None
        self.loss_function = None
        self.train_data = None
        self.test_data = None
        self.train_data_loader = None
        self.test_data_loader = None
        self.model_name = ""
        self.optim_name = "SGD"
        self.test_data_len = 0
        self.train_data_len = 0
        self.acc = []
        self.loss = []
        self.logger = logger
        self.test_acc = []
        self.test_loss = []
        self.args = None
        self.scheduler = None
        # self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.INFO)
        
        # # 创建控制台处理器，输出到控制台
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        
        # # 创建文件处理器，输出到文件
        # log_filename = f'./logs/activate_cifar_training_{time.strftime("%Y_%m_%d_%H_%M_%S")}.log'
        # file_handler = logging.FileHandler(log_filename)
        # file_handler.setLevel(logging.INFO)
        
        # # 创建日志格式
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # console_handler.setFormatter(formatter)
        # file_handler.setFormatter(formatter)
        
        # # 将处理器添加到logger
        # self.logger.addHandler(console_handler)
        # self.logger.addHandler(file_handler)

    def model_load_A(self, model_name, input,dim_out,args):
        self.model_name = model_name
        if (model_name == "Lenetcifar"):
            self.model = Lenetcifar(dim_out)
        elif (model_name == "CNNcifar"):
            self.model = CNNcifar(input,dim_out)
        elif (model_name == "resnet18"):
            # self.model = resnet18(pretrained=False, num_classes=1000)
            self.model = resnet18()
            if args.datasets=="cifar100":
                self.model.num_classes=100 
            else:
                self.model.num_classes=10
        elif (model_name == "resnet34"):
            # self.model = resnet34(pretrained=False, num_classes=1000)
            self.model = resnet34()
            if args.datasets=="cifar100":
                self.model.num_classes=100 
            else:
                self.model.num_classes=10
        elif (model_name == "resnet50"):
            # self.model = resnet50(pretrained=False, num_classes=1000)
            self.model = resnet50()
            if args.datasets=="cifar100":
                self.model.num_classes=100 
            else:
                self.model.num_classes=10
        # elif (model_name == "resnet101"):
        #     self.model = resnet101(pretrained=False, num_classes=1000)
        #     if args.datasets=="cifar100":
        #         self.model.num_classes=100 
        #     else:
        #         self.model.num_classes=10
        # elif (model_name == "resnet152"):
        #     self.model = resnet101(pretrained=False, num_classes=1000)
        #     if args.datasets=="cifar100":
        #         self.model.num_classes=100 
        #     else:
        #         self.model.num_classes=10
        else:
            self.model = MLPCifar(32 * input * 3, 100, dim_out)
        # self.logger.info("model name", self.model)
        self.logger.info("active_model_name: {}".format(self.model_name))
        self.logger.info("active_model_num_classes: {}".format(self.model.num_classes))
        # self.model = MLP(28*14, 100, 10)
        # self.model = CNNMnist()
        # self.model = Lenetmnist()
        # print(self.model)
        # if model_name in ["resnet18", "resnet34", "resnet50","resnet101","resnet152"]:      
        #     self.model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
        #     self.model.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
        #     num_ftrs = self.model.fc.in_features  # 获取（fc）层的输入的特征数
        #     # self.logger.info("active_model_num_classes: {}".format(self.model.num_classes))
        #     self.model.fc = nn.Linear(num_ftrs, self.model.num_classes)
        return self.model
    

    def optimizer_load_A(self, model):
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,momentum=0.9)
        lambda1 = lambda ep: 0.99 ** ep
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        return self.optimizer

    def loss_funtion_load_A(self):
        self.loss_function = nn.CrossEntropyLoss()
        return self.loss_function

    # print("train_data")
    def data_load(self, BATCH_SIZE):
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=False,num_workers=4)
        self.test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.args.batch_size * 10, shuffle=False,num_workers=4)
        self.train_data_len = len(self.train_data_loader)
        self.test_data_len = len(self.test_data_loader)
        self.train_data_loader = iter(self.train_data_loader)
        self.test_data_loader = iter(self.test_data_loader)


    def data_load_train(self, train_data_loader, labels):
        # self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=BATCH_SIZE, shuffle=False)
        # self.test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=BATCH_SIZE * 10, shuffle=False)
        self.train_data_len = len(train_data_loader)
        #self.test_data_len = len(self.test_data_loader)
        self.train_data_loader = train_data_loader
        self.labels = labels
        #self.test_data_loader = iter(self.test_data_loader)

    def data_load_test(self, test_data_loader, labels):
        self.test_data_len = len(test_data_loader)
        self.test_data_loader = test_data_loader
        self.labels = labels


    # def forward_train_1(self, model):
    #    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     img = self.train_data_loader
    #     #self.labels = labels
    #     img.to(device)
    #    # self.labels.to(device)
    #     # img,_ = self.train_data_loader[i]
    #     x1 = model(img)
    #     return x1
    def forward_train_1(self, model):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img, labels = next(self.train_data_loader)
        self.labels = labels
        img = img.to(device)
        self.labels.to(device)
        # img,_ = self.train_data_loader[i]
        x1 = model(img)
        return x1
    
    def forward_test_1(self, model):
        img, labels = next(self.test_data_loader)
        img, self.labels = img.to(device), labels.to(device)
        x1 = model(img)
        return x1
    
    # def forward_test_1(self, model):
    #     img = self.test_data_loader
    #     #self.labels = labels
    #     img = img.to(device)
    #     x1 = model(img)
    #     return x1

    def feature_aggregation(self, inter):
        n = len(inter)
        # print("n")
        # print(n)
        # out = inter[0]
        # out1 = inter[0]
        # for i in inter[1:]:
        #     out1 = torch.cat((out1, i), 1)
        out1 = sum(inter) / n
        # loss = self.loss_function(out1, label)
        return out1

    def forward_train_2(self, model, train_x, training):
        train_x = train_x.to(device)
        out = model.forward2(train_x, training)
        return out

    def loss_aggregation(self, inter):
        n = len(inter)
        out1 = sum(inter) / n
        return out1

    def loss_calculate(self, out):
        self.labels = self.labels.to(device)
        loss = self.loss_function(out, self.labels)
        return loss

    def loss_calculate_test(self, out):
        loss = self.loss_function(out, self.labels)
        total = self.labels.size(0)
        return loss

    def acc_calculate_test(self, predicted):
        self.labels = self.labels.to(device)
        num = (predicted == self.labels).sum().item()
        total = self.labels.size(0)
        return (100 * num) / total



    def backward_A(self, optimizer, loss):
        loss.backward(retain_graph=True)

    def weight_aggregation(self, local_embedding, local_pre):
        weight = []
        for pre in local_pre:
            # print("self.acc_calculate_test(pre)",self.acc_calculate_test(pre))
            if self.acc_calculate_test(pre) != 0:
                weight.append(self.acc_calculate_test(pre))
            else:
                weight.append(0.001)
        w = sum(weight)
        global_embedding = []
        for i in range(len(local_embedding)):
            ge = (weight[i] / w) * local_embedding[i]
            global_embedding.append(ge)
        return sum(global_embedding)
    
    def AVG_aggregation(self, local_embedding, local_pre):
        w = sum(local_embedding)
        n = len(local_embedding)
        return (1/n)*w


    def pre_train(self, args):
        # model_name = arg.model
        # model = self.model_load_A(model_name, 8)
        # self.optimizer = self.optimizer_load_A(self.model)
        # self.loss_function = self.loss_funtion_load_A()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(device)
        
        epoch_loss = []
        epoch_acc = []
        train_results = []
        test_results = []
        
        
        
        for ep in range(args.local_ep):
            batch_loss = []
            batch_acc = []
            self.data_load(BATCH_SIZE=args.batch_size)
            running_loss = 0.0
            running_acc = 0
            
            # train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=128, shuffle=False)
            self.data_load(args.batch_size)
            
            # 每轮训练
            for i, (img, label) in enumerate(self.train_data_loader, 0):
                self.model.train()
                self.optimizer.zero_grad()
                # train_dataset, split_train_data = split_vertical_data_all_cifar10_1(args, img)
                # img = copy.deepcopy(split_train_data[0])
                img = img.to(device)
                label = label.to(device)
                
                x1 = self.model(img)
                out = self.model.forward2(x1, 'True')
                self.labels = label
                loss = self.loss_calculate(out)
                
                self.backward_A(self.optimizer, loss)
                self.optimizer.step()
                
                _, predicted = torch.max(out.data, 1)
                total = label.size(0)  # labels 的长度
                label = label.to(device)
                correct = (predicted == label).sum().item()  # 预测正确的数目
                running_acc = 100 * correct / total
                running_loss = loss.item()
                
                batch_loss.append(running_loss)
                batch_acc.append(running_acc)
            
            # 训练结果（每轮训练后输出）
            train_loss = sum(batch_loss) / len(batch_loss)
            train_acc = sum(batch_acc) / len(batch_acc)
            # print(f'[Training {ep+1}/{arg.rounds}], loss: {train_loss:.3f}, accuracy: {train_acc:.3f}')
            self.logger.info(f'[activate Training {ep}/{args.rounds}], loss: {train_loss:.3f}, accuracy: {train_acc:.3f}')
            
            
            # 保存训练结果
            train_results.append([ep, train_loss, train_acc])

            # 每轮训练后进行一次测试
            # print(f'Start Testing after epoch {ep + 1}')
            self.logger.info(f'Start Testing after epoch {ep}')
            test_loss, test_acc = self.test(args)
            
            # 保存测试结果
            test_results.append([ep, test_loss, test_acc])
        
        df_train = pd.DataFrame(train_results, columns=['Epoch', 'Training Loss', 'Training Accuracy'])
        df_test = pd.DataFrame(test_results, columns=['Epoch', 'Test Loss', 'Test Accuracy'])

        # Get current timestamp for file naming
        timestamp = time.strftime('%Y_%m_%d_%H_%M_%S')

        # Format the file name with the provided arguments and timestamp
        file_name = './save/localepoch_new/{}/PVFed_localepoch_{}_pre_local_active_{}_{}_{}_{}_{}_{}_train1.xlsx'.format(
            args.datasets, 
            args.local_ep, 
            args.datasets, 
            self.model_name, 
            args.rounds, 
            args.aggre, 
            args.epsilon, 
            timestamp
        )

        # Create folder if it does not exist
        folder_name = f'./save/localepoch_new/{args.datasets}'
        os.makedirs(folder_name, exist_ok=True)

        # Save training and testing results to an Excel file with two sheets
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            # Write training results to the 'Training Results' sheet
            df_train.to_excel(writer, sheet_name='Training Results', index=False)
            # Write testing results to the 'Test Results' sheet
            df_test.to_excel(writer, sheet_name='Test Results', index=False)

        self.logger.info(f"Results have been saved to {file_name}")

        

    def test(self, args):
        """
        测试过程
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()  # 切换到评估模式
        correct = 0
        total = 0
        test_loss = 0.0
        self.model.eval()
        # test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=128*10, shuffle=False)
        
        with torch.no_grad():
            for i, (img, label) in enumerate(self.test_data_loader, 0):
                # for img, label in test_data_loader:
                img, label = img.to(device), label.to(device)
                
                # test_dataset, split_test_data = split_vertical_data_all_cifar10_1(args, img)
                # img = copy.deepcopy(split_test_data[0])
                
                x1 = self.model(img)
                out = self.model.forward2(x1, "False")
                
                self.labels = label
                loss = self.loss_calculate_test(out)
                test_loss += loss.item()
                
                # 计算预测准确率
                _, predicted = torch.max(out.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        # 计算并返回测试损失和准确率
        test_loss /= len(self.test_data_loader)
        test_acc = 100 * correct / total
        # print(f'[Test], loss: {test_loss:.3f}, accuracy: {test_acc:.3f}')
        self.logger.info(f'[Test], loss: {test_loss:.3f}, accuracy: {test_acc:.3f}')
        return test_loss, test_acc
    
    # def train(self, arg):
    #     model_name = arg.model
    #     model = self.model_load_A(model_name, 8)
    #     self.optimizer = self.optimizer_load_A(self.model)
    #     self.loss_function = self.loss_funtion_load_A()
    #     epoch_loss = []
    #     epoch_acc =[]
    #     model.train()
    #     for ep in range(arg.rounds):
    #         batch_loss = []
    #         batch_acc = []
    #         self.data_load(BATCH_SIZE=arg.batch_size)
    #         running_loss = 0.0
    #         running_acc = 0
    #         n = 0
    #         for i, (img, label) in enumerate(self.train_data_loader, 0):
    #             train_dataset, split_train_data = split_vertical_data_all_cifar10_1(args, img)
    #             img = copy.deepcopy(split_train_data[0])
    #             self.optimizer.zero_grad()
    #             x1 = self.model(img)
    #             inter = []
    #             re = []
    #             inter.append(x1)
    #             train_x = self.feature_aggregation(inter)
    #             out = self.model.forward2(train_x, 'True')
    #             self.labels = label
    #             loss = self.loss_calculate(out)
    #             self.backward_A(self.optimizer, loss)
    #             self.optimizer.step()
    #             _, predicted = torch.max(out.data, 1)
    #             total = label.size(0)  # labels 的长度
    #             correct = (predicted == label).sum().item()  # 预测正确的数目
    #             running_acc = 100 * correct / total
    #             running_loss = loss.item()
    #             batch_loss.append(running_loss)
    #             batch_acc.append(running_acc)
    #         print('[active_training{:d},{:d}], loss: {:.3f}, accure : {:.3f}.'.format(ep, args.local_ep, 
    #                                                                                     sum(batch_loss)/len(batch_loss),
    #                                                                                     sum(batch_acc)/len(batch_acc)))
    #         epoch_acc.append(sum(batch_acc)/len(batch_acc))
    #         epoch_loss.append(sum(batch_loss)/len(batch_loss))
    #     print('Finished Training')
    #     test_acc = []
    #     test_loss = []
    #     print('Start Testing')
    #     correct = 0
    #     loss = 0
    #     total = 0
    #     model.eval()
    #     i = 0
    #     running_loss = 0
    #     with torch.no_grad():
    #         for data in self.test_data_loader:
    #             imgs, labels = data
    #             test_dataset, split_test_data = split_vertical_data_all_cifar10_1(args, imgs)
    #             imgs = split_test_data[0]
    #             x1 = self.model(imgs)
    #             self.labels = labels
    #             out = self.model.forward2(x1, "False")
    #             loss = self.loss_calculate_test(out)
    #             _, predicted = torch.max(out.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #             print('[testing{:d},{:d}], loss: {:.3f}, accure : {:.3f}.'.format(arg.rounds, i,
    #                                                                                   loss.item(),
    #                                                                                   100 * correct / total))
    #             test_loss.append(loss.item())
    #             test_acc.append(100 * correct / total)
    #             i = i + 1
    #     epoch_acc.append(0)
    #     epoch_loss.append(0)
    #     epoch_acc.extend(test_acc)
    #     epoch_loss.extend(test_loss)
    #     data_dict = {'acc': epoch_acc, 'loss': epoch_loss}
    #     print('train_acc', epoch_acc)
    #     print('train_loss', epoch_loss)
    #     print('test loss', test_loss)
    #     print('test acc', test_acc)

    #     df = pd.DataFrame.from_dict(data_dict)
        
    #     folder_name = 'save/localepoch/{}'.format(args.datasets)
    #     create_folder(folder_name)
    #     df.to_excel(
    #         './save/localepoch/{}/PVFed_localepoch_{}_pre_local_active_{}_{}_{}_{}_{}_{}_train1.xlsx'.format(args.datasets, args.local_ep, args.datasets,
    #                                                             self.model_name,
    #                                                             args.rounds, args.aggre, args.epsilon, str(time.strftime('%Y_%m_%d_%H_%M_%S'))))

if __name__ == "__main__":
    args = args_parser()
    data = Data(args)
    # train_dataset, A_train_dataset, A_test_dataset, B_test_dataset, B_train_dataset = data.split_vertical_data(args)
    train_dataset, test_dataset = data.load_dataset(args)
    #train_dataset, test_dataset, split_train_data, split_test_data = data.split_vertical_data_all(args)
    active = active_cifar()
    active.data_load_train(train_dataset.data, train_dataset.labels)
    # print(train_dataset.shape)
    active.train(args)
