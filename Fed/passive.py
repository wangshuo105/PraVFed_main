import os

import torch.optim
import tqdm as tqdm
from torch import nn
from torch.utils.data import DataLoader

from models.models import *
from options import args_parser
from data.data_process import Data
from models.resnet import resnet18, resnet34, resnet101, resnet50,resnet152
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


def modify_resnet_for_mnist(pretrained_model, num_classes=10):
    # 获取预训练的 ResNet 模型
    model = pretrained_model

    # 修改第一层卷积层的输入通道数
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 通道数从3改为1

    # 修改全连接层的输出类别数（MNIST 有10个类别）
    model.fc = nn.Linear(512 * model.block.expansion, num_classes)

    return model

class passive(object):
    def __init__(self, B_train_dataset, B_test_dataset,logger):
        self.model = None
        #self.data_loader = data_loader
        self.global_params = None
        self.optimizer = None
        self.loss_function = None
        self.train_data = B_train_dataset
        self.test_data = B_test_dataset
        self.embedding = None
        self.train_data_loader = None
        self.test_data_loader = None
        self.optim_name = "SGD"
        self.model_name = ""
        self.acc = []
        self.loss = []
        self.inter_result=[]
        
        self.logger = logger



    def model_load_B(self, model_name, input_dim, out_dim,args):
        self.model_name = model_name
        if (model_name == "MLP"):
            self.model = MLP(28 * input_dim, 100, out_dim)
        elif (model_name == "CNNMnist"):
            self.model = CNNMnist(out_dim)
        elif (model_name == "Lenetmnist"):
            self.model = Lenetmnist(out_dim)
        elif (model_name == "resnet18"):
            self.model = resnet18(pretrained=False, channel=1, num_classes=10)
        elif (model_name == "resnet34"):
            self.model = resnet34(pretrained=False, channel=1, num_classes=10)
        elif (model_name == "resnet50"):
            self.model = resnet50(pretrained=False, channel=1, num_classes=10)
        elif (model_name == "resnet101"):
            self.model = resnet101(pretrained=False, channel=1, num_classes=10)
        else:
            print("no model name")
        # print(model_name)
        # print("model_load_B",device)
        self.logger.info("passive_model_name: {}".format(self.model_name))
        self.model.to(device)
        return self.model
    
    def model_load_B_adult(self, model_name, input_dim, out_dim):
        self.model_name = model_name
        self.model = MLP_adult(input_dim,out_dim)
        return self.model

    def optimizer_load_B(self, model):
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        return self.optimizer


    def optimizer_load_B_optim(self, model, optim_name):
        if (optim_name == "SGD"):
            self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        elif(optim_name == "SGD_moment"):
            self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8, nesterov=True)
        elif(optim_name == "Adam"):
            self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
        else:
            self.optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
        self.optim_name = optim_name
        print(self.optim_name)
        return self.optimizer

    def loss_funtion_load_B(self):
        self.loss_function = nn.CrossEntropyLoss()
        return self.loss_function

    def data_load(self, BATCH_SIZE):
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=BATCH_SIZE, shuffle=False)
        self.test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=BATCH_SIZE * 10, shuffle=False)
        self.train_data_loader = iter(self.train_data_loader)
        self.test_data_loader = iter(self.test_data_loader)


    def forward_train_1(self, model):
        img, _ = next(self.train_data_loader)
        img = img.to(device)
        # model.to(device)
        x1 = model(img)
        return x1

    def forward_test_1(self, model):
        img, _ = next(self.test_data_loader)
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
        x1 = model.No_Train_forward(img)
        return x1

    def No_train_forward_train_2(self, model, train_x, training=False):
        out = model.forward2(train_x, training)
        return out
    
    def No_train_forward_test_1(self, model):
        img, _ = next(self.test_data_loader)
        img = img.to(device)
        #model.to(device)
        x1 = model.No_Train_forward(img)
        return x1

    def No_train_forward_test_2(self, model, train_x, training=False):
        out = model.forward2(train_x, training)
        return out

    def loss_calculate(self, out, labels):
        loss = self.loss_function(out, labels)
        return loss
    def backward_B(self,optimizer, loss):
        #optimizer.zero_grad()
        loss.backward(retain_graph=True)
        #loss.detach()

    def pre_train(self, args, k):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"passive device: {device}")
        
        # 初始化存储训练和测试结果的列表
        epoch_loss = []
        epoch_acc = []
        train_results = []
        test_results = []
        class_number = 9
        # 获取训练数据加载器
        # self.data_load(BATCH_SIZE=128)
        
        # 训练过程
        for ep in range(args.local_ep):
            running_loss = 0.0
            running_acc = 0
            batch_loss = []
            batch_acc = []
            
            self.data_load(args.batch_size)
            # 训练阶段
            for i, (img, label) in enumerate(self.train_data_loader, 0):
                # 标签加噪声
                # labels = add_noise_with_dp(label, args.epsilon)
                
                img, label = img.to(device), label.to(device)
                noisy_label = add_noise_with_dp(label, args.epsilon, args.out_dim-1)
                label = noisy_label
                
                self.model.train()
                self.optimizer.zero_grad()
                # print(img.shape)
                # print("label",label.shape)
                # if img.dim() == 2:
                #     img = img.unsqueeze(1)
                #     img = img.view(-1, 1, 2, 2)
                #     print(img.shape)
                # 前向传播
                x1 = self.model(img)
                out = self.model.forward2(x1, 'True')
                loss = self.loss_calculate(out, label)
                
                # 反向传播
                self.backward_B(self.optimizer, loss)
                self.optimizer.step()
                
                # 计算准确率
                _, predicted = torch.max(out.data, 1)
                total = label.size(0)
                correct = (predicted == label).sum().item()
                running_acc = 100 * correct / total
                running_loss = loss.item()
                
                # 保存每个batch的损失和准确率
                batch_loss.append(running_loss)
                batch_acc.append(running_acc)
            
            # 计算并保存每轮的训练损失和准确率
            train_loss = sum(batch_loss) / len(batch_loss)
            train_acc = sum(batch_acc) / len(batch_acc)
            # print(f'[Training {ep+1}/{args.local_ep}], loss: {train_loss:.3f}, accuracy: {train_acc:.3f}')
            self.logger.info(f'[Passive_Training {ep}/{args.local_ep}], loss: {train_loss:.3f}, accuracy: {train_acc:.3f}')
            # 存储训练结果
            train_results.append([ep, train_loss, train_acc])

            # 每轮训练后进行一次测试
            # print(f'Start Testing after epoch {ep + 1}')
            # self.logger.info(f'Passive Start Testing after epoch {ep}')
            test_loss, test_acc = self.test(args,ep)
            
            # 存储测试结果
            test_results.append([ep + 1, test_loss, test_acc])

        # 将训练和测试结果保存到Excel文件
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

        print(f"Results have been saved to {file_name}")

        # print('Finished Training and Testing')
        self.logger.info('Finished Training and Testing')

    def test(self, args,ep):
        """
        测试过程
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()  # 切换到评估模式
        correct = 0
        total = 0
        test_loss = 0.0
        class_number = 9
        
        with torch.no_grad():
            for img, label in self.test_data_loader:
                img, label = img.to(device), label.to(device)
                x1 = self.model(img)
                out = self.model.forward2(x1, "False")
                noisy_label = add_noise_with_dp(label, args.epsilon, args.out_dim-1)
                label = noisy_label
                
                loss = self.loss_calculate(out, label)
                test_loss += loss.item()
                
                # 计算预测准确率
                _, predicted = torch.max(out.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        # 计算并返回测试损失和准确率
        test_loss /= len(self.test_data_loader)
        test_acc = 100 * correct / total
        # print(f'[Passive_Test: {ep}], loss: {test_loss:.3f}, accuracy: {test_acc:.3f}')
        self.logger.info(f'[Passive_Test: {ep}], loss: {test_loss:.3f}, accuracy: {test_acc:.3f}')
        # self.logger.info(f'[Test], loss: {test_loss:.3f}, accuracy: {test_acc:.3f}')
        return test_loss, test_acc

#     def pre_train(self, args,i):
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         print(device)
#         epoch_loss = []
#         epoch_acc = []
# #        self.model.train()
#         for ep in range(args.local_ep):
#             # startTick = time.clock()
#             self.data_load(BATCH_SIZE=128)
#             running_loss = 0.0
#             running_acc = 0
#             batch_loss = []
#             batch_acc = []
#             n = 0
#             for i, (img, label) in enumerate(self.train_data_loader, 0):
#                 labels = add_noise_with_dp(label,args.epsilon)
#                 img, label = img.to(device),labels.to(device)
#                 self.model.train()
#                 self.optimizer.zero_grad()
#                 x1 = self.model(img)
#                 out = self.model.forward2(x1, 'True')
#                 loss = self.loss_calculate(out, label)
#                 self.backward_B(self.optimizer, loss)
#                 self.optimizer.step()
#                 _, predicted = torch.max(out.data, 1)
#                 total = label.size(0)  # labels 的长度
#                 correct = (predicted == label).sum().item()  # 预测正确的数目
#                 running_acc = 100 * correct / total
#                 running_loss = loss.item()
#                 batch_loss.append(running_loss)
#                 batch_acc.append(running_acc)
#             print('[active_training{:d},{:d}], loss: {:.3f}, accure : {:.3f}.'.format(ep, args.local_ep, 
#                                                                                         sum(batch_loss)/len(batch_loss),
#                                                                                         sum(batch_acc)/len(batch_acc)))
#             epoch_acc.append(sum(batch_acc)/len(batch_acc))
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
#         print('Finished Training')
#         #self.pre_acc = train_acc
#         test_acc = []
#         test_loss = []
#         print('Start Testing')
#         correct = 0
#         loss = 0
#         total = 0
#         self.model.eval()
#         i = 0
#         running_loss = 0
#         with torch.no_grad():
#             for data in self.test_data_loader:
#                 images, labels = data
#                 images, labels = images.to(device), labels.to(device)
#                 x1 = self.model(images)
#                 labels = add_noise_with_dp(labels,args.epsilon)
#                 #self.labels = labels
#                 out = self.model.forward2(x1, "False")
#                 loss = self.loss_calculate(out, labels)
#                 _, predicted = torch.max(out.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 print('[testing{:d},{:d}], loss: {:.3f}, accure : {:.3f}.'.format(args.local_ep, i,
#                                                                                       loss.item(),
#                                                                                       100 * correct / total))
#                 test_loss.append(loss.item())
#                 test_acc.append(100 * correct / total)
#                 i = i + 1
#                 # print('Accuracy of the network on the test images: %.3f %%, loss: %.3f %%' % (
#                 # 100.0 * correct / total),loss.item())
#         epoch_acc.append(0)
#         epoch_loss.append(0)
#         epoch_acc.extend(test_acc)
#         epoch_loss.extend(test_loss)
#         data_dict = {'acc': epoch_acc, 'loss': epoch_loss}
#         print('train_acc', epoch_acc)
#         print('train_loss', epoch_loss)
#         print('test loss', test_loss)
#         print('test acc', test_acc)

#         df = pd.DataFrame.from_dict(data_dict)
    
#         folder_name = 'save/localepoch/{}'.format(args.datasets)
#         create_folder(folder_name)
#         df.to_excel(
#             './save/localepoch/{}/PVFed_localepoch_{}_pre_local_passive_{}_{}_{}_{}_{}_{}_train1.xlsx'.format(args.datasets, args.local_ep, args.datasets,
#                                                                 self.model_name,
#                                                                 args.rounds, args.aggre, args.epsilon, str(time.strftime('%Y_%m_%d_%H_%M_%S'))))






if __name__ == "__main__":

    args = args_parser()
    data = Data(args)
    # train_dataset, A_train_dataset, A_test_dataset, B_test_dataset, B_train_dataset = data.split_vertical_data(args)
    # train_dataset, test_dataset = data.load_dataset(args)
    train_dataset, test_dataset, split_train_data, split_test_data = data.split_vertical_data_all(args)
    passive = passive(split_train_data[3], split_test_data[3])
    # print(train_dataset.shape)
    passive.pre_train(args, "MLP")