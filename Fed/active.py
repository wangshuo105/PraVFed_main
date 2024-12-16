# 1. 本地模型前向训练
# 2. 聚合本地模型参数
# 3. 将本地模型参数发送给其他客户端
# 4. 聚合最终的结果，并进行损失函数计算
# 5. 将损失函数传输给其他客户端
# 6. 本地模型参数更新
# 7. 联合预算准确率
import time

import torch.optim
import tqdm as tqdm
from torch import nn
from torch.utils.data import DataLoader

from models.models import *
from models.resnet import resnet18, resnet34, resnet101, resnet50,resnet152
from options import args_parser
from data.data_process import Data
import pandas as pd
from data.data_process import *
import time
import os
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_folder(folder_name):
    try:
        os.makedirs(folder_name)
        #print(f"Fold{folder_name} finish")
    except FileExistsError:
        print(f"f folder {folder_name} exist")






class active(object):
    def __init__(self, A_train_dataset, A_test_dataset,logger):
        self.labels = None
        self.model = None
        # self.data_loader = data_loader
        self.global_params = None
        self.optimizer = None
        self.loss_function = None
        self.train_data = A_train_dataset
        self.test_data = A_test_dataset
        self.train_data_loader = None
        self.test_data_loader = None
        self.optim_name = "SGD"
        self.model_name = ""
        self.test_data_len = 0
        self.train_data_len = 0
        self.acc = []
        self.loss = []
        self.test_acc = []
        self.test_loss = []
        self.logger = logger
        

    def model_load_A(self, model_name, input,out_dim,args):
        self.model_name = model_name
        if (model_name == "MLP"):
            self.model = MLP(28 * input, 100, out_dim)
        elif (model_name == "CNNMnist"):
            self.model = CNNMnist(out_dim)
        elif (model_name == "Lenetmnist"):
            self.model = Lenetmnist(out_dim)
        elif (model_name == "resnet18"):
            self.model = resnet18(pretrained=True, channel=1, num_classes=10)
        elif (model_name == "resnet34"):
            self.model = resnet34(pretrained=True, channel=1, num_classes=10)
        elif (model_name == "resnet50"):
            self.model = resnet50(pretrained=True, channel=1, num_classes=10)
        elif (model_name == "resnet101"):
            self.model = resnet101(pretrained=True, channel=1, num_classes=10)
        else:
            self.logger.info("no model name")
        self.logger.info("active_model_name: {}".format(self.model_name))
        return self.model


    def model_load_A_adult(self, model_name, input_dim, out_dim):
        self.model_name = model_name
        self.model = MLP_adult(input_dim,out_dim)
        return self.model
    
    def optimizer_load_A(self, model):
        #self.optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        self.optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
        return self.optimizer

    def loss_funtion_load_A(self):
        self.loss_function = nn.CrossEntropyLoss()
        return self.loss_function

    # print("train_data")

    def data_load(self, BATCH_SIZE):
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=BATCH_SIZE, shuffle=False)
        self.test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=BATCH_SIZE * 10, shuffle=False)
        self.train_data_len = len(self.train_data_loader)
        self.test_data_len = len(self.test_data_loader)
        self.train_data_loader = iter(self.train_data_loader)
        self.test_data_loader = iter(self.test_data_loader)

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



    def weight_aggregation(self, local_embedding, local_pre):
        weight = []
        for pre in local_pre:
            # num = (pre == label).sum().item()
            # total = label.size(0)
            # weight.append((100 * num) / total)
            weight.append(self.acc_calculate_test(pre))
        # out1 = sum(inter) / n
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

    def forward_train_2(self, model, train_x, training):
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
    
    def loss_calculate1(self, out, label):
        loss = self.loss_function(out, label)
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

    # def loss_calculate_test(self, out, labels):
    #     loss = self.loss_function(out, labels)
    #     return loss

    def backward_A(self, optimizer, loss):
        # self.optimizer.zero_grad()
        # torch.autograd.set_detect_anamoly()
        loss.backward(retain_graph=True)
        # optimizer.step()
    def pre_train(self, args):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(device)
        
        # 初始化存储训练和测试结果的列表
        epoch_loss = []
        epoch_acc = []
        train_results = []
        test_results = []
        
        # 获取训练数据加载器
        
        
        # 训练过程
        for ep in range(args.local_ep):
            running_loss = 0.0
            running_acc = 0
            batch_loss = []
            batch_acc = []
            
            self.data_load(BATCH_SIZE=128)
            # 每轮训练
            for i, (img, label) in enumerate(self.train_data_loader, 0):
                img, label = img.to(device), label.to(device)
                self.model.train()
                self.optimizer.zero_grad()
                
                # 前向传播
                x1 = self.model(img)
                out = self.model.forward2(x1, 'True')
                loss = self.loss_calculate1(out, label)
                
                # 反向传播
                self.backward_A(self.optimizer, loss)
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
            self.logger.info(f'[Active Training {ep}/{args.local_ep}], loss: {train_loss:.3f}, accuracy: {train_acc:.3f}')
            
            # 存储训练结果
            train_results.append([ep, train_loss, train_acc])

            # 每轮训练后进行一次测试
            # print(f'Start Testing after epoch {ep + 1}')
            self.logger.info(f'Start Activate Testing after epoch {ep}')
            test_loss, test_acc = self.test(args,ep)
            
            # 存储测试结果
            test_results.append([ep + 1, test_loss, test_acc])

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

        # print('Finished Training and Testing')

    def test(self, args,ep):
        """
        测试过程
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()  # 切换到评估模式
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for img, label in self.test_data_loader:
                img, label = img.to(device), label.to(device)
                x1 = self.model(img)
                out = self.model.forward2(x1, "False")
                
                loss = self.loss_calculate1(out, label)
                test_loss += loss.item()
                
                # 计算预测准确率
                _, predicted = torch.max(out.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        # 计算并返回测试损失和准确率
        test_loss /= len(self.test_data_loader)
        test_acc = 100 * correct / total
        # print(f'[active_Test: {ep}], loss: {test_loss:.3f}, accuracy: {test_acc:.3f}')
        self.logger.info(f'[active_Test: {ep}], loss: {test_loss:.3f}, accuracy: {test_acc:.3f}')
        return test_loss, test_acc
#     def pre_train(self, args):
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         print(device)
#         # train_loss = []
#         # train_acc = []
#         epoch_loss = []
#         epoch_acc = []
# #        self.model.train()
#         for ep in range(args.local_ep):
#             # startTick = time.clock()
#             batch_loss = []
#             batch_acc = []
#             self.data_load(BATCH_SIZE=128)
#             running_loss = 0.0
#             running_acc = 0
#             n = 0
#             for i, (img, label) in enumerate(self.train_data_loader, 0):
#                 img, label = img.to(device),label.to(device)
#                 self.model.train()
#                 self.optimizer.zero_grad()
#                # print("img",img.is_cuda, "model",self.model.is_cuda)
#                 x1 = self.model(img)
#                 out = self.model.forward2(x1, 'True')
#                 loss = self.loss_calculate1(out,label)
#                 self.backward_A(self.optimizer, loss)
#                 self.optimizer.step()
#                 _, predicted = torch.max(out.data, 1)
#                 total = label.size(0)  # labels 的长度
#                 correct = (predicted == label).sum().item()  # 预测正确的数目
#                 running_acc = 100 * correct / total
#                 # loss 的输出，每个一百个batch输出，平均的loss
#                 running_loss = loss.item()
#                 batch_loss.append(running_loss)
#                 batch_acc.append(running_acc)
#             print('[active_training{:d},{:d}], loss: {:.3f}, accure : {:.3f}.'.format(ep, args.local_ep, 
#                                                                                         sum(batch_loss)/len(batch_loss),
#                                                                                         sum(batch_acc)/len(batch_acc)))
#             epoch_acc.append(sum(batch_acc)/len(batch_acc))
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#         print('Finished Training')
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
#                 #self.labels = labels
#                 out = self.model.forward2(x1, "False")
#                 loss = self.loss_calculate1(out, labels)
#                 # log_probs, protos = model(images)
#                 _, predicted = torch.max(out.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 #running_loss += loss.item()
#                 #if i % 10 == 9:
#                     # running_loss += loss.item()
#                 print('[testing{:d},{:d}], loss: {:.3f}, accure : {:.3f}.'.format(args.local_ep, i,
#                                                                                       loss.item(),
#                                                                                       100 * correct / total))
#                 test_loss.append(loss.item())
#                 test_acc.append(100 * correct / total)
#                 i = i + 1
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
#             './save/localepoch/{}/PVFed_localepoch_{}_pre_local_active_{}_{}_{}_{}_{}_{}_train1.xlsx'.format(args.datasets, args.local_ep, args.datasets,
#                                                                 self.model_name,
#                                                                 args.rounds, args.aggre, args.epsilon, str(time.strftime('%Y_%m_%d_%H_%M_%S'))))


if __name__ == "__main__":
    args = args_parser()
    data = Data(args)
    # train_dataset, A_train_dataset, A_test_dataset, B_test_dataset, B_train_dataset = data.split_vertical_data(args)
    #train_dataset, test_dataset = data.load_dataset(args)
    train_dataset, test_dataset, split_train_data, split_test_data = data.split_vertical_data_all(args)
    active_party = active(split_train_data[0], split_test_data[0])
    # print(train_dataset.shape)
    active_model_name = "MLP"
    input_dim = 7
    active_party.model = active_party.model_load_A(active_model_name, input_dim)
    active_party.model.to(device)
    active_party.optimizer = active_party.optimizer_load_A(active_party.model)
    active_party.loss_function = active_party.loss_funtion_load_A()
    active_party.pre_train(args)
