import os

import torch.optim
import tqdm as tqdm
from torch import nn
from torch.utils.data import DataLoader

from models.models import Lenetcifar, CNNcifar, MLPCifar
from options import args_parser
from data.data_process import *
from models.resnet_cifar import *
import pandas as pd
from data.RDP import add_noise_with_dp
import time
import logging

def create_folder(folder_name):
    try:
        os.makedirs(folder_name)
        #print(f"Fold{folder_name} finish")
    except FileExistsError:
        print(f"f folder {folder_name} exist")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class passive_cifar(object):
    def __init__(self,logger):
        self.model = None
        #self.data_loader = data_loader
        self.global_params = None
        self.optimizer = None
        #self.loss_function = None
        self.train_data = None
        self.test_data = None
        self.optim_name = "SGD"
        self.train_data_loader = None
        self.test_data_loader = None
        self.model_name = ""
        self.acc = []
        self.loss = []
        self.logger = logger
        self.embedding = None
        self.args = None
       

    def model_load_B(self, model_name, input, dim_out,args):
        self.model_name = model_name
        if (model_name == "Lenetcifar"):
            self.model = Lenetcifar(dim_out)
        elif (model_name == "CNNcifar"):
            self.model = CNNcifar(input,dim_out)
        elif (model_name == "resnet18"):
            self.model = resnet18()
            if args.datasets=="cifar100":
                self.model.num_classes=100 
            else:
                self.model.num_classes=10
        elif (model_name == "resnet34"):
            self.model = resnet34()
            if args.datasets=="cifar100":
                self.model.num_classes=100 
            else:
                self.model.num_classes=10
        elif (model_name == "resnet50"):
            self.model = resnet50()
            if args.datasets=="cifar100":
                self.model.num_classes=100 
            else:
                self.model.num_classes=10
        else:
            self.model = MLPCifar(32 * input * 3, 100, dim_out)
        self.logger.info("passive_model_name: {}".format(self.model_name))
        # print(self.model)
        return self.model

    def optimizer_load_B(self, model):
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,momentum=0.9)
        lambda1 = lambda ep: 0.99 ** ep
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        return self.optimizer

    def optimizer_load_B_optim(self, model, optim_name):
        if (optim_name == "SGD"):
            self.optimizer = torch.optim.SGD(model.parameters(), lr = self.args.lr)
        elif(optim_name == "SGD_moment"):
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        elif(optim_name == "Adam"):
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, betas=(0.9, 0.99))
        else:
            self.optimizer = torch.optim.Adagrad(model.parameters(), lr=self.args.lr)
        self.optim_name = optim_name
        
        print(self.optim_name)
        return self.optimizer

    def loss_funtion_load_B(self):
        self.loss_function = nn.CrossEntropyLoss()
        return self.loss_function

    def data_load_train(self, train_data_loader):
        self.train_data_len = len(train_data_loader)
        #self.test_data_len = len(self.test_data_loader)
        self.train_data_loader = train_data_loader
        #self.test_data_loader = iter(self.test_data_loader)
        
    def data_load(self, train_data):
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=False,num_workers=4)
        self.test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.args.batch_size * 10, shuffle=False,num_workers=4)
        self.train_data_loader = iter(self.train_data_loader)
        self.test_data_loader = iter(self.test_data_loader)

    def data_load_test(self, test_data_loader):
        self.test_data_len = len(test_data_loader)
        self.test_data_loader = test_data_loader
        return self.test_data_loader

    def forward_train_1(self, model):
        img = self.train_data_loader
        img = img.to(device)
        x1 = model(img)
        return x1

    def forward_test_1(self, model):
        img = self.test_data_loader
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img = img.to(device)
        x1 = model(img)
        return x1

    def forward_train_2(self, model, train_x, training):
        out = model.forward2(train_x, training)
        return out

    def No_train_forward_train_1(self, model):
        img, _ = next(self.train_data_loader)
        img = img.to(device)
        #model.to(device)
        x1 = model(img)
        return x1

    def No_train_forward_train_2(self, model, train_x, training=False):
        out = model.forward2(train_x, training)
        return out
    
    
    def No_train_forward_test_2(self, model, train_x, training=False):
        out = model.forward2(train_x, training)
        return out
    
    
    def No_train_forward_test_1(self, model):
        img, _ = next(self.test_data_loader)
        img = img.to(device)
        #model.to(device)
        x1 = model(img)
        return x1

    
    def loss_calculate(self, out, labels):
        loss = self.loss_function(out, labels)
        return loss

    def backward_B(self,optimizer, loss):
        #optimizer.zero_grad()
        loss.backward(retain_graph=True)
        #loss.detach()

   

    def pre_train(self, args, k):
        epoch_loss = []
        epoch_acc = []
        epoch_test_loss = []
        epoch_test_acc = []
        data = Data(args)
 
        df_columns = ['Epoch', 'Training Loss', 'Training Accuracy']
        train_results = []
        
        df_columns_test = ['Epoch', 'Test Loss', 'Test Accuracy']
        test_results = []
        
        if args.datasets == "cifar100":
            class_number = 99
        else:
            class_number = 9

        for ep in range(args.local_ep):
            batch_loss = []
            batch_acc = []
            running_loss = 0.0
            running_acc = 0.0
            n = 0
            
        
            
            self.data_load(args.batch_size)
            # 训练过程
            self.model.train()
            for i, (img, label) in enumerate(self.train_data_loader, 0):
                img, label = img.to(device), label.to(device)
                label
                # self.labels = label
                # train_dataset, split_train_data = split_vertical_data_all_cifar10_1(args, img)
                # print("class_number",class_number)
                noisy_label = add_noise_with_dp(label, args.epsilon, class_number)
                label = noisy_label
                
                # self.data_load_train(split_train_data[k])
                # self.data_load_train(img)
                self.optimizer.zero_grad()
                # self.train_data_loader = self.train_data_loader.to(device)
                # x1 = self.model(self.train_data_loader)
                # print(img.shape)
                # # img = img.squeeze(1)
                # print(img.shape)
                x1 = self.model(img)
                
                out = self.model.forward2(x1, 'True')
                # print("label",label.shape)
                loss = self.loss_calculate(out, label)
                
                self.backward_B(self.optimizer, loss)
                self.optimizer.step()
                
                _, predicted = torch.max(out.data, 1)
                total = label.size(0)  # labels 的长度
                correct = (predicted == label).sum().item()  # 预测正确的数目
                running_acc = 100 * correct / total
                running_loss = loss.item()
                batch_loss.append(running_loss)
                batch_acc.append(running_acc)
        
            self.scheduler.step()
            # active_party.scheduler.step()
            # 每轮训练后，计算并输出训练结果
            train_loss = sum(batch_loss) / len(batch_loss)
            train_acc = sum(batch_acc) / len(batch_acc)
            self.logger.info(f'[Training {ep}/{args.local_ep}], loss: {train_loss:.3f}, accuracy: {train_acc:.3f}')
            
            epoch_acc.append(train_acc)
            epoch_loss.append(train_loss)
            
            # 将训练损失和准确率保存到 train_results
            train_results.append([ep, train_loss, train_acc])
            
            # 测试过程
            test_loss, test_acc = self.test(args, self.test_data, k,class_number)
            
            # 将测试结果保存到 test_results
            test_results.append([ep, test_loss, test_acc])
            
            df_train = pd.DataFrame(train_results, columns=['Epoch', 'Training Loss', 'Training Accuracy'])
            df_test = pd.DataFrame(test_results, columns=['Epoch', 'Test Loss', 'Test Accuracy'])

            # Get current timestamp for file naming
            if ep == 0:
                timestamp = time.strftime('%Y_%m_%d_%H_%M_%S')

            # Format the file name with the provided arguments and timestamp
            file_name = './save/localepoch_new/{}/PVFed_localepoch_{}_pre_local_passive_{}_{}_{}_{}_{}_{}_train1.xlsx'.format(
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
            
            
        # self.logger.info('Finished Training and Testing')

    def test(self, args, test_dataset, k,class_number):
        """
        测试过程
        """
        self.model.eval()  # 切换到评估模式
        test_loss = 0.0
        correct = 0
        total = 0
        # train_dataset, test_dataset = Data(args).load_dataset(args)  # 只获取测试数据集
        # test_data_loader_all = torch.utils.data.DataLoader(test_dataset, batch_size=10*args.batch_size, shuffle=False)
        
        with torch.no_grad():
            for i, (img, label) in enumerate(self.test_data_loader, 0):
            # for img, label in self.test_data_loader:
                img, label = img.to(device), label.to(device)
                # train_dataset, split_test_data = split_vertical_data_all_cifar10_1(args, img)
                # self.test_data_loader = self.data_load_test(split_test_data[k])
                noisy_label = add_noise_with_dp(label, args.epsilon, class_number)
                label = noisy_label

                # 使用当前的模型进行预测
                # x1 = self.model(self.test_data_loader)
                x1 = self.model(img)
                out = self.model.forward2(x1, 'False')
                # print("out",out.shape)
                # print("label",label.shape)
                loss = self.loss_calculate(out, label)
                test_loss += loss.item()

                _, predicted = torch.max(out.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        # 计算并返回测试损失和准确率
        test_loss /= len(self.test_data_loader)
        test_acc = 100 * correct / total
        # print(f'[Passive_Test {k}], loss: {test_loss:.3f}, accuracy: {test_acc:.3f}')
        self.logger.info(f'[Test], loss: {test_loss:.3f}, accuracy: {test_acc:.3f}')
        return test_loss, test_acc






if __name__ == "__main__":

    args = args_parser()
    # data = Data(args)
   # train_dataset, A_train_dataset, A_test_dataset, B_test_dataset, B_train_dataset = data.split_vertical_data(args)
    passive_party = passive_cifar()
    model_name = "Lenetcifar"
    passive_party.model = passive_party.model_load_B(model_name, 7)
    #passive.model.to(device)
    passive_party.optimizer = passive_party.optimizer_load_B(passive_party.model)
    passive_party.loss_function = passive_party.loss_funtion_load_B()
    passive_party.pre_train(args)