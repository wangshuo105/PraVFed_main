import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # VFL
    parser.add_argument('--rounds', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--local_ep', type=int, default=2, help="number of rounds of local training")
    parser.add_argument('--num_user', type=int, default=4, help="number of party(both active and passive): K")
    parser.add_argument('--frac', type=float, default=0.04, help='the fraction of clients: C')
    parser.add_argument('--local_bs', type=int, default=4, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--central", type=str, default='single',help="training model type")
    # model
    parser.add_argument('--model', type=str, default='Lenetmnist', help="model name")
    parser.add_argument('--model_type', type=str, default='hoto', help="mode")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imgs")
    parser.add_argument('--out_dim', type=int, default=2, help="out of channels of imgs")
    parser.add_argument('--max_pool', type=str, default='True', help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--epsilon', type=float, default=0.5,help="ep number")
    parser.add_argument('--aggre',type=str,default='weightagg',help="aggregation type")
    # data
    parser.add_argument('--datasets',type=str, default='fmnist', help="dataset type")
    #parser.add_argument('--train', type=str, default='True', help="the dataset is or not training")
    # parser.add_argument('--split', type=int, default=392, help="split imgs feature")
    parser.add_argument('--batch_size', type=int, default=128, help="local batch_size")
    parser.add_argument('--shuffle', type=str, default='True',help="whether shuffle the dataset")
    parser.add_argument('--nThreads',type=int, default=10, help="number of threads when read data")
    # other
    parser.add_argument('--data_dir', type=str, default='../data/', help="directory of dataset")
    #parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="name of classes")
    parser.add_argument('--optimizer', type=str, default='SGD', help="type of optimizer")
    parser.add_argument('--seed', type=int, default=1234, help="random seed")
    parser.add_argument('--test_ep', type=int, default=10, help="num of test episodes for evaluation")
    parser.add_argument('--agg', type=str, default='False', help="whether use the agg out")
    parser.add_argument('--optim', type=str, default='False', help="whether use the agg out")

    args = parser.parse_args()
    return args

