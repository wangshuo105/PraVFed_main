import os

import pandas as pd
import torch
from data.data_process import *
from Fed.active import active
from Fed.passive import passive
from options import args_parser
from random import choice
from Fed.active_cifar import active_cifar
from Fed.passive_cifar import passive_cifar
import openpyxl
import time
import random
import numpy as np
from Fed.lib import *
from xlsxwriter import Workbook
import logging
# from data.adult_data_process import *

def create_folder(folder_name):
    try:
        os.makedirs(folder_name)
        #print(f"Fold{folder_name} finish")
    except FileExistsError:
        print(f"f folder {folder_name} exist")

def setup_logger(args):
    """
    设置日志记录器，输出日志到控制台和文件
    """
    # Ensure the 'logs' directory exists
    log_dir = './logs'
    create_folder(log_dir)

    logger = logging.getLogger("PraVFed_Logger")
    logger.setLevel(logging.INFO)


    logger.info(args)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建文件处理器，文件路径包括时间戳
    log_filename = os.path.join(log_dir, f'PraVFed_train_{args.datasets}_{args.model}_{args.rounds}_{args.aggre}_{args.local_ep}_{time.strftime("%Y_%m_%d_%H_%M_%S")}.log')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def model_select(args, datasets,tabular_datasets,img32_datasets):
    
    if args.datasets in datasets or args.datasets in tabular_datasets:
        if args.model == "MLP" or args.model == "CNNMnist" or args.model == "Lenetmnist":
            if args.model_type == "homo":
                model_name_all = [args.model]
            else:
                model_name_all = ["Lenetmnist", "CNNMnist", "MLP"]
        else:
            if args.model_type == "homo":
                model_name_all = [args.model]
            else:
                model_name_all = ["resnet18", "resnet34", "resnet50"]
        if args.datasets in tabular_datasets:
            input_dim = 4
            args.out_dim = 2
        else:
            input_dim = 7
            args.out_dim = 10
    elif args.datasets in img32_datasets:
        if args.model == "MLPCifar" or args.model == "CNNcifar" or args.model == "Lenetcifar":
            if args.model_type == "homo":
                model_name_all = [args.model]
            else:
                model_name_all = ["Lenetcifar", "CNNcifar", "MLPCifar"]
        else:
            if args.model_type == "homo":
                model_name_all = [args.model]
            else:
                model_name_all = ["resnet18", "resnet34", "resnet50"]
        input_dim = 8
        args.out_dim = 100 if args.datasets == "cifar100" else 10
    else:
        # print("Dataset not supported!")
        logger.info("Dataset not supported!")
    
    return model_name_all, input_dim, args.out_dim
    


def PraVFed_train(args,logger):
    # 数据加载
    
    datasets = ["mnist","fmnist"]
    img32_datasets = ["cifar10","cifar100","cinic"]
    tabular_datasets=["adult","breast","diabetes","creditcard"]
    # == "mnist" or args.datasets == "fmnist" or args.datasets == "adult"
    data = Data(args)
    
    if (args.datasets in datasets) or (args.datasets in tabular_datasets):
        if args.datasets in tabular_datasets:
            if args.datasets == "adult":
                train_dataset, test_dataset, split_train_data, split_test_data = adult_data(args.num_user)
            elif args.datasets == "diabetes":
                train_dataset, test_dataset, split_train_data, split_test_data = diabetes_data(args.num_user)
            elif args.datasets == "creditcard":
                train_dataset, test_dataset, split_train_data, split_test_data = creditcard_data(args.num_user)
            else:
                train_dataset, test_dataset, split_train_data, split_test_data = breast_cancer(args.num_user,args)
        else:
            train_dataset, test_dataset, split_train_data, split_test_data = data.split_vertical_data_all(args)
        active_party = active(split_train_data[0], split_test_data[0],logger)
    else:
        train_dataset, test_dataset = data.load_dataset(args)
        active_party = active_cifar(logger)
        train_dataset_split= split_data_2(train_dataset,args.num_user)
        test_dataset_split= split_data_2(test_dataset,args.num_user)
        active_party.train_data = train_dataset_split[0]
        active_party.test_data = test_dataset_split[0]
        active_party.args = args
    P = []
    

    for i in range(args.num_user - 1):
        if args.datasets in datasets or args.datasets in tabular_datasets:
            logger.info(f"data i, {i}")
            passive_party = passive(split_train_data[i + 1], split_test_data[i + 1],logger)
        else:
            passive_party = passive_cifar(logger)
            passive_party.train_data = train_dataset_split[i+1]
            passive_party.test_data = test_dataset_split[i+1]
            passive_party.args = args
        P.append(passive_party)
    
    model_name_all, input_dim, out_dim = model_select(args,datasets,tabular_datasets,img32_datasets)
    

    for i in range(len(P)):
        passive_party = P[i]
        model_name = model_name_all[i % len(model_name_all)]  
        # model_name = choice(model_name_all)  
        # print("model_name",model_name)
        logger.info(f"model_name: {model_name}")
        if args.datasets in tabular_datasets:
            input_dim = passive_party.train_data.tensors[0].shape[1]
            # print("input_dim",input_dim,out_dim)
            passive_party.model = passive_party.model_load_B_adult(model_name, input_dim, out_dim)
        else:
            passive_party.model = passive_party.model_load_B(model_name, input_dim, out_dim, args)
        passive_party.model.to(device)
        passive_party.optimizer = passive_party.optimizer_load_B(passive_party.model)
        passive_party.loss_function = passive_party.loss_funtion_load_B()
        passive_party.pre_train(args, i + 1)
    
    # 加载主动方模型
    if args.datasets in tabular_datasets:
        input_dim = active_party.train_data.tensors[0].shape[1]
        print("input_dim",input_dim)
        active_party.model = active_party.model_load_A_adult(args.model, input_dim, out_dim)
    else:
        active_party.model = active_party.model_load_A(args.model, input_dim, out_dim, args)
    
    active_party.model.to(device)
    active_party.optimizer = active_party.optimizer_load_A(active_party.model)
    active_party.loss_function = active_party.loss_funtion_load_A()
    # active_party.pre_train(args)
    
    # 创建文件存储每轮训练和测试的结果
    folder_name = f'save/PraVFed_Main/{args.datasets}/{args.model_type}'
    create_folder(folder_name)
    
    # Excel 文件路径
    excel_file_path = f'./save/PraVFed_Main/{args.datasets}/{args.model_type}/PVFed_two_{args.datasets}_{active_party.model_name}_{args.rounds}_{args.aggre}_{args.epsilon}_{time.strftime("%Y_%m_%d_%H_%M_%S")}.xlsx'
    communications = []
    # 使用 pandas 写入 Excel 文件
    with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
        
        # 训练轮次
        for ep in range(args.rounds):
            batch_acc = []
            batch_loss = []
            # if args.datasets in datasets or args.datasets in tabular_datasets:
            active_party.data_load(args.batch_size)
            for passive_party in P:
                passive_party.data_load(args.batch_size)
            train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4)
            epoch_communication = 0
            # lambda1 = lambda ep: 0.7 ** ep
            # scheduler = torch.optim.lr_scheduler.LambdaLR(active_party.optimizer, lr_lambda=lambda1)
            # active_party.scheduler.step()
            # 训练过程
            for i, (img, label) in enumerate(train_data_loader, 0):
                # 分割数据，适应不同数据集的分割方法
                label = label.to(device)
                # active_party.labels = label
                
                local_embedding = []
                local_pre = []
                
                # 被动方处理
                for p, passive_party in enumerate(P):
                    # passive_party.model.eval()
                    x1 = passive_party.No_train_forward_train_1(passive_party.model)
                    passive_comm = tensor_size_in_bit(x1) if ep == 0 else 0
                    local_embedding.append(x1)
                    pre = passive_party.No_train_forward_train_2(passive_party.model, x1, training=False)
                    _, predicted = torch.max(pre.data, 1)
                    passive_comm += tensor_size_in_bit(pre)
                    epoch_communication += passive_comm
                    local_pre.append(predicted)
                
                # 主动方处理
                active_party.model.train()
                active_party.optimizer.zero_grad()
                x1_a = active_party.forward_train_1(active_party.model)
                # print(local_pre)
                if args.aggre == "weightagg":
                    global_embedding = active_party.weight_aggregation(local_embedding, local_pre)
                else:
                    global_embedding = active_party.AVG_aggregation(local_embedding, local_pre)
                global_embedding = global_embedding.to(device)
                # all_x = (global_embedding + x1_a) / 2
                all_x = global_embedding + x1_a
                out_a = active_party.forward_train_2(active_party.model, all_x, "True")
                loss_a = active_party.loss_calculate(out_a)
                active_party.backward_A(active_party.optimizer, loss_a)
                _, predicted_a = torch.max(out_a.data, 1)
                active_party.optimizer.step()
                # active_party.scheduler.step()
                
                # 计算训练准确率
                correct_a = (predicted_a == label).sum().item()
                total = label.size(0)
                batch_loss.append(loss_a.item())
                batch_acc.append(100 * correct_a / total)
            # for p, passive_party in enumerate(P):
            #     passive_party.scheduler.step()
            # active_party.scheduler.step()
            # 打印当前轮次训练结果
            # logger.info(f'Training A:[{ep+1},{args.rounds}], loss: {sum(batch_loss)/len(batch_loss):.3f}, accuracy: {sum(batch_acc)/len(batch_acc):.3f}, Communication: {epoch_communication / (1024 * 1024):.3f} MB')
            active_party.loss.append(sum(batch_loss) / len(batch_loss))
            active_party.acc.append(sum(batch_acc) / len(batch_acc))
            if ep == 0:
                communication = epoch_communication / (1024 * 1024)
            else:
                communication = communications[-1] + epoch_communication / (1024 * 1024)
                
            communications.append(communication)  
            logger.info(f'Training A:[{ep+1},{args.rounds}], loss: {sum(batch_loss)/len(batch_loss):.3f}, accuracy: {sum(batch_acc)/len(batch_acc):.3f}, Communication: {communication:.3f} MB')

            # 每一轮训练后将训练结果写入Excel文件
            df_train = pd.DataFrame({
                'Epoch': [ep+1], 
                'Training Accuracy': [sum(batch_acc)/len(batch_acc)], 
                'Training Loss': [sum(batch_loss)/len(batch_loss)], 
                'Communication': [f'{communication:.3f}']
            })
            if ep == 0:
                df_train.to_excel(writer, sheet_name='Training Results', index=False, startrow=ep, header=True)
            else:
                # 否则只写入数据
                df_train.to_excel(writer, sheet_name='Training Results', index=False, startrow=ep+1, header=False)
            # df_train.to_excel(writer, sheet_name='Training Results', index=False, startrow=ep+1, header=(ep==0))
            
            # df_train.to_excel(writer, sheet_name='Training Results', index=False, startrow=ep+1, header=(ep==0))


            test_acc, test_loss = test_model(active_party, P, test_dataset, img32_datasets, args, device, logger,ep)

            df_test = pd.DataFrame({
                'Epoch': [ep+1], 
                'Test Accuracy': [test_acc], 
                'Test Loss': [test_loss]
            })
            if ep == 0:
                df_test.to_excel(writer, sheet_name='Testing Results', index=False, startrow=ep, header=True)
            else:
                # 否则只写入数据
                df_test.to_excel(writer, sheet_name='Testing Results', index=False, startrow=ep+1, header=False)
        
            # df_test.to_excel(writer, sheet_name='Testing Results', index=False, startrow=ep+1, header=(ep==0))
        if ep % 5 == 0:
            args.lr = args.lr / 5
        
        if args.rounds >= 5:
            last_5_train_acc = active_party.acc[-5:]
            last_5_test_acc = active_party.test_acc[-5:]
        else:
            last_5_train_acc = active_party.acc[-1:]
            last_5_test_acc = active_party.test_acc[-1:]
        avg_train_acc = np.mean(last_5_train_acc)
        avg_test_acc = np.mean(last_5_test_acc)
        std_train_acc = np.std(last_5_train_acc)
        std_test_acc = np.std(last_5_test_acc)
        
        df_train = pd.DataFrame({
                'Epoch': [args.rounds+1], 
                'Training Accuracy': [avg_train_acc], 
                'Training Loss': [std_train_acc], 
                'Communication': [0]
            })
        
        logger.info(f'Traing A:[{args.rounds+1},{args.rounds}], loss: {std_train_acc}, accuracy: {avg_train_acc}')
        df_train.to_excel(writer, sheet_name='Training Results', index=False, startrow=args.rounds+1, header=False)
        
        df_test = pd.DataFrame({
                'Epoch': [args.rounds+1], 
                'Test Accuracy': [avg_test_acc], 
                'Test Loss': [std_test_acc]
        })
        logger.info(f'Testing A:[{args.rounds+1},{args.rounds}], loss: {std_test_acc}, accuracy: {avg_test_acc}')
        df_test.to_excel(writer, sheet_name='Testing Results', index=False, startrow=args.rounds+1, header=False)
            
        

    logger.info(f"Results saved to: {excel_file_path}")


def test_model(active_party, P, test_dataset, img32_datasets, args, device, logger, ep):
    """
    测试模型的函数，进行评估并返回测试结果。
    """
    test_acc = []
    test_loss = []
    correct = 0
    loss = 0
    total = 0

    # 使用 torch.no_grad() 禁用梯度计算
    with torch.no_grad():
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size * 10, shuffle=True)
        # active_party.data_load(args.batch_size)
        # for passive_party in P:
        #     passive_party.data_load(args.batch_size)
        for i, (img, label) in enumerate(test_data_loader, 0):
            active_party.model.eval()
            # 存储每个 party 的预测结果
            local_embedding = []
            local_pre = []

            # 被动方的测试
            for p, passive_party in enumerate(P):
                passive_party.model.eval()
                x1 = passive_party.No_train_forward_test_1(passive_party.model)
                local_embedding.append(x1)
                pre = passive_party.No_train_forward_test_2(passive_party.model, x1, training=False)
                _, predicted = torch.max(pre.data, 1)
                local_pre.append(predicted)

            # 主动方的测试
            x1_a = active_party.forward_test_1(active_party.model)
            if args.aggre == "weightagg":
                global_embedding = active_party.weight_aggregation(local_embedding, local_pre)
            else:
                global_embedding = active_party.AVG_aggregation(local_embedding, local_pre)
            all_x = global_embedding + x1_a
            out_a = active_party.forward_train_2(active_party.model, all_x, "False")
            loss_a = active_party.loss_calculate(out_a)
            _, predicted_a = torch.max(out_a.data, 1)
            correct_a = active_party.acc_calculate_test(predicted_a)

            # 保存结果
            test_loss.append(loss_a.item())
            test_acc.append(correct_a)

        # 输出测试结果
        avg_test_loss = sum(test_loss) / len(test_loss)
        avg_test_acc = sum(test_acc) / len(test_acc)
        logger.info(f'[Testing Epoch: {ep+1}], loss: {avg_test_loss}, accuracy: {avg_test_acc}')
        active_party.test_loss.append(avg_test_loss)
        active_party.test_acc.append(avg_test_acc)
        
    return avg_test_acc, avg_test_loss

if __name__ == '__main__':
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = args_parser()
    logger = setup_logger(args)
    logger.info(device)
    logger.info("命令行参数:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    PraVFed_train(args,logger)