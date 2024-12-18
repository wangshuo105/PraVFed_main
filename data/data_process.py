import copy

import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from options import args_parser
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from uuid import uuid4
from torch.utils.data import Dataset, DataLoader
import random


class Data(object):
    def __init__(self, args):
        self.data_loader = {}
        self.num_steps = 0
        # nthreads denoted the read data
        self.nThreads = args.nThreads
        # self.transform = transform
        self.batch_size = args.batch_size
        # split data into two parts
        self.shuffle = args.shuffle
        # self.split_vertical_data(args)

    def load_dataset(self, args):
        print(args.datasets)
        global train_dataset, test_dataset
        if args.datasets == "mnist":
            trans_mnist_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = torchvision.datasets.MNIST("./data/mnist", train=True,
                                                       transform=trans_mnist_train,
                                                       download=True)
            test_dataset = torchvision.datasets.MNIST("./data/mnist", train=False,
                                                      transform=trans_mnist_train,
                                                      download=True)
        elif args.datasets == "fmnist":
            trans_Fmnist_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = torchvision.datasets.FashionMNIST("data/Fmnist", train=True,
                                                              transform=trans_Fmnist_train,
                                                              download=True)
            test_dataset = torchvision.datasets.FashionMNIST("data/Fmnist", train=False,
                                                             transform=trans_Fmnist_train,
                                                             download=True)
        elif args.datasets == "cifar10":
            # trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
            #                                           transforms.RandomHorizontalFlip(),
            #                                           transforms.ToTensor(),
            #                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                                                std=[0.229, 0.224, 0.225])])
            trans_cifar10_train  = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                  transforms.RandomCrop(32, padding=4),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                                                                    ])
            trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
            train_dataset = torchvision.datasets.CIFAR10("data/cifar10", train=True,
                                                         transform=trans_cifar10_train,
                                                         download=True)
            test_dataset = torchvision.datasets.CIFAR10("data/cifar10", train=False,
                                                        transform=trans_cifar10_val,
                                                        download=True)
        elif args.datasets == "cifar100":
            trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                           std=[0.229, 0.224, 0.225])])
            trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
            train_dataset = torchvision.datasets.CIFAR100("data/cifar100", train=True,
                                                         transform=trans_cifar100_train,
                                                         download=True)
            test_dataset = torchvision.datasets.CIFAR100("data/cifar100", train=False,
                                                        transform=trans_cifar100_val,
                                                        download=True)
        elif args.datasets == "cinic":
            trans_cinic_train = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                                                    # transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], 
                                                                        std=[0.24205776, 0.23828046, 0.25874835])  # CINIC数据集的标准化
                                                    ])
            trans_cinic_val = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], 
                                                                    std=[0.24205776, 0.23828046, 0.25874835])  # CINIC数据集的标准化
                                                ])
            train_dataset_train = datasets.ImageFolder(root='data/cinic-10/train', transform=trans_cinic_train)
            samples = train_dataset_train.imgs
            random.shuffle(samples)
            train_dataset_train.imgs = samples
            train_dataset_valid = datasets.ImageFolder(root='data/cinic-10/valid', transform=trans_cinic_train)
            samples = train_dataset_valid.imgs
            random.shuffle(samples)
            train_dataset_valid.imgs = samples
            train_dataset = ConcatDataset([train_dataset_train, train_dataset_valid])
            test_dataset = datasets.ImageFolder(root='data/cinic-10/test', transform=trans_cinic_val)
            samples = test_dataset.imgs
            random.shuffle(samples)
            test_dataset.imgs = samples
        else:
            print("Please input dataset type")
        # shuffle dataset manually
        if (self.shuffle):
            if ((args.datasets == "mnist") or (args.datasets == "Fmnist")):
                size = len(train_dataset.data)
                idx = np.random.choice(size, size, replace=False)
                # print(idx)
                train_dataset.data = train_dataset.data[idx]
                # print(train_dataset.targets)
                train_dataset.targets = train_dataset.targets[idx]

        return train_dataset, test_dataset
    
    
    def split_data(self,dataset, n_workers=4):
        # if worker_list is None:
        worker_list = list(range(0, n_workers))

        # counter to create the index of different data samples
        idx = 0

        # dictionary to accomodate the split data
        dic_single_datasets = {}
        for worker in worker_list:
            """
            Each value is a list of three elements, to accomodate, in order: 
            - data examples (as tensors)
            - label
            - index 
            """
            dic_single_datasets[worker] = []

        """
        Loop through the dataset to split the data and labels vertically across workers. 
        Splitting method from @abbas5253: https://github.com/abbas5253/SplitNN-for-Vertically-Partitioned-Data/blob/master/distribute_data.py
        """
        label_list = []
        index_list = []
        index_list_UUID = []
        for tensor, label in dataset:
            height = tensor.shape[-1] // len(worker_list)
            i = 0
            uuid_idx = uuid4()
            for worker in worker_list[:-1]:
                dic_single_datasets[worker].append(torch.unsqueeze(tensor[:, :, height * i: height * (i + 1)], 0))
                i += 1

            # add the value of the last worker / split
            
            dic_single_datasets[worker_list[-1]].append(torch.unsqueeze(tensor[:, :, height * (i):], 0))
            label_list.append(torch.Tensor([label]))
            index_list_UUID.append(uuid_idx)
            index_list.append(torch.Tensor([idx]))

            idx += 1
        
        
        for worker in worker_list:
            l = len(dic_single_datasets[worker])
            data_tempt = torch.cat(dic_single_datasets[worker][:l])
            dic_single_datasets[worker] = data_tempt
        l = len(label_list)
        label = torch.cat(label_list[:l])
        # print(label.shape)
        
        return dic_single_datasets, label, index_list, index_list_UUID
    
    def split_data_1(self, dataset, n_workers):
        worker_list = list(range(0, n_workers))

        dic_single_datasets = {}
        for worker in worker_list:
            dic_single_datasets[worker] = []

        label_list = []
        idx = 0
        for tensor, label in dataset:
            height = tensor.shape[-1] // len(worker_list)
            i = 0
            for worker in worker_list[:-1]:

                dic_single_datasets[worker].append((torch.unsqueeze(tensor[:, :, height * i: height * (i + 1)], 0), label))
                i += 1
            dic_single_datasets[worker_list[-1]].append((torch.unsqueeze(tensor[:, :, height * i:], 0), label))

            label_list.append(label)
            idx += 1

        for worker in worker_list:
            data_tempt = torch.cat([data[0] for data in dic_single_datasets[worker]])  
            labels_tempt = [data[1] for data in dic_single_datasets[worker]]  
            dic_single_datasets[worker] = (data_tempt, torch.tensor(labels_tempt))  
        for worker in dic_single_datasets:
            data, labels = dic_single_datasets[worker]
            print(f"Worker {worker}: Data shape = {data.shape}, Labels shape = {labels.shape}")
        return dic_single_datasets
    
    
    
    
    def split_vertical_data(self, args):
        # split dataset into two A_dataset and B_dataset
        train_dataset, test_dataset = self.load_dataset(args)
        train_dataset_all = copy.deepcopy(train_dataset)
        A_train_dataset = copy.deepcopy(train_dataset)
        B_train_dataset = copy.deepcopy(train_dataset)
        A_test_dataset = copy.deepcopy(test_dataset)
        B_test_dataset = copy.deepcopy(test_dataset)
        train_data_dim = train_dataset.data.shape[2]
        # print(train_data_dim)
        split_dim = int(train_data_dim / 2)
        # print(split_dim)
        # print(train_dataset.data.shape)
        A_train_data, B_train_data = torch.split(train_dataset.data, split_dim, dim=2)
        A_test_data, B_test_data = torch.split(test_dataset.data, split_dim, dim=2)
        A_train_dataset.data = copy.deepcopy(A_train_data)
        A_test_dataset.data = copy.deepcopy(A_test_data)
        B_train_dataset.data = copy.deepcopy(B_train_data)
        B_test_dataset.data = copy.deepcopy(B_test_data)
        return train_dataset_all, A_train_dataset, A_test_dataset, B_test_dataset, B_train_dataset

    def split_vertical_data_all(self, args):
        split_train_data = []
        split_test_data = []
        train_dataset, test_dataset = self.load_dataset(args)
        train_dataset_all = copy.deepcopy(train_dataset)
        test_dataset_all = copy.deepcopy(test_dataset)
        N = args.num_user
        data_dim = train_dataset.data.shape[2]
        split_data_dim = int(data_dim / N)
        split_train = torch.split(train_dataset.data, split_data_dim, dim=2)
        split_test = torch.split(test_dataset.data, split_data_dim, dim=2)
        # final_split_train = []
        if (len(split_train) > N):
            for i in range(N - 1):
                inter = copy.deepcopy(train_dataset_all)
                inter.data = copy.deepcopy(split_train[i])
                split_train_data.append(inter)
                inter_test = copy.deepcopy(test_dataset_all)
                inter_test.data = copy.deepcopy(split_test[i])
                split_test_data.append(inter_test)
            inter = torch.cat([split_train[-2], split_train[-1]], dim=2)
            inter_train = copy.deepcopy(train_dataset_all)
            inter_train.data = copy.deepcopy(inter)
            split_train_data.append(inter_train)

            inter = torch.cat([split_test[-2], split_test[-1]], dim=2)
            inter_test = copy.deepcopy(test_dataset_all)
            inter_test.data = copy.deepcopy(inter)
            split_test_data.append(inter_test)
        else:
            for i in range(N):
                inter = copy.deepcopy(train_dataset_all)
                inter.data = copy.deepcopy(split_train[i])
                split_train_data.append(inter)
                inter_test = copy.deepcopy(test_dataset_all)
                inter_test.data = copy.deepcopy(split_test[i])
                split_test_data.append(inter_test)
        return train_dataset, test_dataset, split_train_data, split_test_data

    def split_vertical_data_all_cifar10(self, args):
        train_dataset, test_dataset = self.load_dataset(args)
        train_dataset_all = copy.deepcopy(train_dataset)
        test_dataset_all = copy.deepcopy(test_dataset)
        N = args.num_user
        worker_list = list(range(0, N))
        list_train_datasets = []
        list_test_datasets = []
        for j in range(len(worker_list)):
            """
            Each value is a list of three elements, to accomodate, in order: 
            - data examples (as tensors)
            - label
            - index 
            """
            list_train_datasets.append([])
            list_test_datasets.append([])

        for tensor, label in train_dataset_all:
            height = tensor.shape[-1] // len(worker_list)
            i = 0
            for worker in worker_list[:-1]:
                list_train_datasets[worker].append(tensor[:, :, height * i: height * (i + 1)])
                i += 1

            # add the value of the last worker / split
            list_train_datasets[worker_list[-1]].append(tensor[:, :, height * (i):])

        for test, label in test_dataset_all:
            height = test.shape[-1] // len(worker_list)
            i = 0
            for worker in worker_list[:-1]:
                list_test_datasets[worker].append(test[:, :, height * i: height * (i + 1)])
                i += 1

            # add the value of the last worker / split
            list_test_datasets[worker_list[-1]].append(tensor[:, :, height * (i):])

        return train_dataset, test_dataset, list_train_datasets, list_test_datasets
    

if __name__ == "__main__":
    args = args_parser()
    data = Data(args)
    train_dataset, test_dataset, list_train_datasets, list_test_datasets = data.split_vertical_data_all_cifar10(args)
    load_data = torch.cat(list_train_datasets[0])
    print(load_data.shape)
    train_data = DataLoader(load_data, batch_size=128, drop_last=True)
    print(train_data.data.shape)

    N = args.num_user
    split_data = []
    list1 = [1, 2, 3, 4, 5]
    list2 = [6, 7, 8, 9, 10]

    # 将列表转换为张量
    tensor1 = torch.tensor(list1)
    tensor2 = torch.tensor(list2)

    # 创建TensorDataset
    dataset = TensorDataset(tensor1, tensor2)

    # 打印数据集的长度
    print("Dataset length:", len(dataset))

    print(dataset[0][0].shape)
    # train_dataset, A_train_dataset, A_test_dataset, B_test_dataset, B_train_dataset = data.split_vertical_data(args)
    # train_dataset, test_dataset, split_train_data, split_test_data = data.split_vertical_data_all_cifar10(args)
    # train_dataset, test_dataset, split_train_data, split_test_data = data.split_vertical_data_all(args)
    # print(train_dataset.data.shape)
    # print(len(split_train_data))
    # print("len", len(split_train_data[0]))
    # print(split_train_data[0][0].shape)
    # #print(split_train_data[0].data.shape())
    # plt_image_all(N, train_dataset, split_train_data)

    # x = torch.randn(2, 3, 8)
    # print(x)
    # print(x.shape)
    # #x1 = x.split([3, 3],dim=2)
    # x1 = torch.split(x, 3, dim=2)
    # x = []
    # for i in range(2):
    #     x.append(x1[i])
    # if len(x1) > 2:
    #     c = torch.cat([x1[-2],x1[-1]],dim=2)
    #     x.append(c)
    # for i in range(len(x)):
    #     print(x[i])
    #     print(x[i].shape)
    # if x1[-1].shape[2] < 3:
    #     for i in range(x1):
    #
    # print(x1[0])
    # print(x1[0].shape)
    # print(x1[1])
    # print(x1[1].shape)
    # print(x1[2])
    # print(x1[2].shape)
    # if x1[-1].shape[2] < 3:
    #     x = torch.cat([x1[-2],x1[-1]],dim=2)
    #     print(x)
    #     print(x.shape)
    #     x1[-2] = tuple(x)
    #    #x1_tuple = tuple(x1)
    # print(x1[-1])
    # print(x1[-1].shape)