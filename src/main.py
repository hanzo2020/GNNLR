# coding=utf-8
#--rank 1 --model_name NPQA --optimizer Adam --lr 0.0001 -- dataset ml100k01-1-5 --metric ndcg@10,precision@1 --max_his 10 --sparse_his 0 --neg_his 1 --l2 1e-4 --r_logic 1e-05 --r_length 1e-4 --random_seed 2022 --train 0 --load 1 --load_model_path NLQ-ML100K.pt --gpu 6
import argparse
import logging
import sys
import numpy as np
import os
import torch
import datetime
import pickle
import copy
import torch_geometric
from utils import utils
from utils.global_p import *
import random
# # import data_loaders
from data_loaders.DataLoader import DataLoader
from data_loaders.ProLogicDL import ProLogicDL

# # import models
from models.BaseModel import BaseModel
from models.NLR import NLR
from models.GNNLR import GNNLR

# # import data processors
from data_processors.DataProcessor import DataProcessor
from data_processors.HistoryDP import HistoryDP
from data_processors.ProLogicDP import ProLogicDP
from data_processors.ProLogicRecDP import ProLogicRecDP

# # import runners
from runners.BaseRunner import BaseRunner


def build_run_environment(para_dict, dl_name, dp_name, model_name, runner_name):
    if type(para_dict) is str:
        para_dict = eval(para_dict)
    if type(dl_name) is str:
        dl_name = eval(dl_name)
    if type(dp_name) is str:
        dp_name = eval(dp_name)
    if type(model_name) is str:
        model_name = eval(model_name)
    if type(runner_name) is str:
        runner_name = eval(runner_name)

    # random seed
    torch.manual_seed(para_dict['random_seed'])
    torch.cuda.manual_seed(para_dict['random_seed'])
    np.random.seed(para_dict['random_seed'])
    torch_geometric.seed(para_dict['random_seed'])

    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = para_dict['gpu']  # default '0'
    logging.info("# cuda devices: %d" % torch.cuda.device_count())

    para_dict['load_data'] = True
    dl_paras = utils.get_init_paras_dict(dl_name, para_dict)
    logging.info(str(dl_name) + ': ' + str(dl_paras))
    data_loader = dl_name(**dl_paras)

    # 需要由data_loader来append_his
    # Need to use data_loader to append_his
    if 'all_his' in para_dict:
        data_loader.append_his(all_his=para_dict['all_his'], max_his=para_dict['max_his'],
                               neg_his=para_dict['neg_his'], neg_column=para_dict['neg_column'])

    # 如果是top n推荐，只保留正例，负例是训练过程中采样得到，并且将label转换为01二值
    # If it's top n recommendation, only keep the positive examples, negative examples are sampled during training, also, convert the label into 0/1 binary values
    if para_dict['rank'] == 1:
        data_loader.label_01()
        if para_dict['drop_neg'] == 1:
            data_loader.drop_neg()

    para_dict['data_loader'] = data_loader
    dp_paras = utils.get_init_paras_dict(dp_name, para_dict)
    logging.info(str(dp_name) + ': ' + str(dp_paras))
    data_processor = dp_name(**dp_paras)

    # # prepare train,test,validation samples need to put before model creation and training, to guarantee for different models but the same random seed, the same testing negative examples are created
    data_processor.get_train_data(epoch=-1, model=model_name)
    data_processor.get_validation_data(model=model_name)
    data_processor.get_test_data(model=model_name)

    # 根据模型需要生成 数据集的特征，特征总共one-hot/multi-hot维度，特征每个field最大值和最小值
    # Generate the dataset features according to the need of the model, features are one-hot/multi-hot dimension, the max and min value of each field of the feature
    features, feature_dims, feature_min, feature_max = \
        data_loader.feature_info(include_id=model_name.include_id,
                                 include_item_features=model_name.include_item_features,
                                 include_user_features=model_name.include_user_features)
    para_dict['feature_num'], para_dict['feature_dims'] = len(features), feature_dims
    para_dict['user_feature_num'] = len([f for f in features if f.startswith('u_')])
    para_dict['item_feature_num'] = len([f for f in features if f.startswith('i_')])
    para_dict['context_feature_num'] = len([f for f in features if f.startswith('c_')])
    data_loader_vars = vars(data_loader)
    for key in data_loader_vars:
        if key not in para_dict:
            para_dict[key] = data_loader_vars[key]
    model_paras = utils.get_init_paras_dict(model_name, para_dict)
    logging.info(str(model_name) + ': ' + str(model_paras))
    model = model_name(**model_paras)
    model.load_model()

    # use gpu
    if torch.cuda.device_count() > 0:
        # model = model.to('cuda:0')
        model = model.cuda()

    # create runner
    runner_paras = utils.get_init_paras_dict(runner_name, para_dict)
    logging.info(str(runner_name) + ': ' + str(runner_paras))
    runner = runner_name(**runner_paras)
    return data_loader, data_processor, model, runner


def main():
    # init args
    init_parser = argparse.ArgumentParser(description='Model', add_help=False)
    init_parser.add_argument('--rank', type=int, default=1,
                             help='1=ranking, 0=rating/click')
    init_parser.add_argument('--data_loader', type=str, default='',
                             help='Choose data_loader')
    init_parser.add_argument('--model_name', type=str, default='BaseModel',
                             help='Choose model to run.')
    init_parser.add_argument('--runner_name', type=str, default='',
                             help='Choose runner')
    init_parser.add_argument('--data_processor', type=str, default='',
                             help='Choose runner')
    init_args, init_extras = init_parser.parse_known_args()

    # choose model
    model_name = eval(init_args.model_name)

    # choose data_loader
    if init_args.data_loader == '':
        init_args.data_loader = model_name.data_loader
    data_loader_name = eval(init_args.data_loader)

    # choose data_processor
    if init_args.data_processor == '':
        init_args.data_processor = model_name.data_processor
    data_processor_name = eval(init_args.data_processor)

    # choose runner
    if init_args.runner_name == '':
        init_args.runner_name = model_name.runner
    runner_name = eval(init_args.runner_name)

    # cmd line paras
    parser = argparse.ArgumentParser(description='')
    parser = utils.parse_global_args(parser)
    parser = data_loader_name.parse_data_args(parser)
    parser = model_name.parse_model_args(parser, model_name=init_args.model_name)
    parser = runner_name.parse_runner_args(parser)
    parser = data_processor_name.parse_dp_args(parser)

    origin_args, extras = parser.parse_known_args()
    args = copy.deepcopy(origin_args)

    # logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(vars(init_args))
    logging.info(vars(origin_args))
    logging.info(extras)

    logging.info('DataLoader: ' + init_args.data_loader)
    logging.info('Model: ' + init_args.model_name)
    logging.info('Runner: ' + init_args.runner_name)
    logging.info('DataProcessor: ' + init_args.data_processor)

    # random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # default '0'
    logging.info("# cuda devices: %d" % torch.cuda.device_count())

    # create data_loader
    args.load_data = True
    dl_para_dict = utils.get_init_paras_dict(data_loader_name, vars(args))
    logging.info(init_args.data_loader + ': ' + str(dl_para_dict))
    data_loader = data_loader_name(**dl_para_dict)

    # 需要由data_loader来append_his
    # Need to use data_loader to append_his
    if 'all_his' in origin_args:
        data_loader.append_his(all_his=origin_args.all_his, max_his=origin_args.max_his,
                               neg_his=origin_args.neg_his, neg_column=origin_args.neg_column)

    # If it's top n recommendation, only keep the positive examples, negative examples are sampled during training, also, convert the label into 0/1 binary values
    if init_args.rank == 1:
        data_loader.label_01()
        if origin_args.drop_neg == 1:
            data_loader.drop_neg()

    # create data_processor
    args.data_loader, args.rank = data_loader, init_args.rank
    dp_para_dict = utils.get_init_paras_dict(data_processor_name, vars(args))
    logging.info(init_args.data_processor + ': ' + str(dp_para_dict))
    data_processor = data_processor_name(**dp_para_dict)

    # # prepare train,test,validation samples need to put before model creation and training, to guarantee for different models but the same random seed, the same testing negative examples are created
    data_processor.get_train_data(epoch=-1, model=model_name)
    data_processor.get_validation_data(model=model_name)
    data_processor.get_test_data(model=model_name)

    # create model
    # Generate the dataset features according to the need of the model, features are one-hot/multi-hot dimension, the max and min value of each field of the feature
    features, feature_dims, feature_min, feature_max = \
        data_loader.feature_info(include_id=model_name.include_id,
                                 include_item_features=model_name.include_item_features,
                                 include_user_features=model_name.include_user_features)
    args.feature_num, args.feature_dims = len(features), feature_dims
    args.user_feature_num = len([f for f in features if f.startswith('u_')])
    args.item_feature_num = len([f for f in features if f.startswith('i_')])
    args.context_feature_num = len([f for f in features if f.startswith('c_')])
    data_loader_vars = vars(data_loader)
    for key in data_loader_vars:
        if key not in args.__dict__:
            args.__dict__[key] = data_loader_vars[key]
    # print(args.__dict__.keys())

    model_para_dict = utils.get_init_paras_dict(model_name, vars(args))
    logging.info(init_args.model_name + ': ' + str(model_para_dict))
    model = model_name(**model_para_dict)

    # init model paras
    model.apply(model.init_paras)

    # use gpu
    if torch.cuda.device_count() > 0:
        # model = model.to('cuda:0')
        model = model.cuda()

    # create runner
    runner_para_dict = utils.get_init_paras_dict(runner_name, vars(args))
    logging.info(init_args.runner_name + ': ' + str(runner_para_dict))
    runner = runner_name(**runner_para_dict)

    # training/testing
    logging.info('Test Before Training: train= %s validation= %s test= %s' % (
        utils.format_metric(
            runner.evaluate(model, data_processor.get_train_data(epoch=-1, model=model), data_processor)),
        utils.format_metric(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor)),
        utils.format_metric(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
        if args.unlabel_test == 0 else '-1') + ' ' + ','.join(runner.metrics))
    # utils.format_metric(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
    # If load > 0, load the model and continue training
    #if args.load > 0:
    #    model.load_model(model_path=args.load_model_path)
    # If train > 0, it means training is needed, otherwise test directly
    if args.train > 0:
        runner.train(model, data_processor)

    # logging.info('Test After Training: train= %s validation= %s test= %s' % (
    #     utils.format_metric(
    #         runner.evaluate(model, data_processor.get_train_data(epoch=-1, model=model), data_processor)),
    #     utils.format_metric(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor)),
    #     utils.format_metric(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
    #     if args.unlabel_test == 0 else '-1') + ' ' + ','.join(runner.metrics))

    # save test results
    train_result = runner.predict(model, data_processor.get_train_data(epoch=-1, model=model), data_processor)
    validation_result = runner.predict(model, data_processor.get_validation_data(model=model), data_processor)
    test_result = runner.predict(model, data_processor.get_test_data(model=model), data_processor)
    np.save(args.result_file.replace('.npy', '__train.npy'), train_result)
    np.save(args.result_file.replace('.npy', '__validation.npy'), validation_result)
    np.save(args.result_file.replace('.npy', '__test.npy'), test_result)
    logging.info('Save Results to ' + args.result_file)

    all_metrics = ['rmse', 'mae', 'auc', 'f1', 'accuracy', 'precision', 'recall']
    if init_args.rank == 1:
        all_metrics = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20', 'ndcg@50', 'ndcg@100'] \
                      + ['hit@1', 'hit@5', 'hit@10', 'hit@20', 'hit@50', 'hit@100'] \
                      + ['precision@1', 'precision@5', 'precision@10', 'precision@20', 'precision@50', 'precision@100'] \
                      + ['recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@50', 'recall@100']
    results = [train_result, validation_result, test_result]
    name_map = ['Train', 'Valid', 'Test']
    datasets = [data_processor.get_train_data(epoch=-1, model=model), data_processor.get_validation_data(model=model)]
    if args.unlabel_test != 1:
        datasets.append(data_processor.get_test_data(model=model))
    for i, dataset in enumerate(datasets):
        metrics = model.evaluate_method(results[i], datasets[i], metrics=all_metrics, error_skip=True)
        log_info = 'Test After Training on %s: ' % name_map[i]
        log_metrics = ['%s=%s' % (metric, utils.format_metric(metrics[j])) for j, metric in enumerate(all_metrics)]
        log_info += ', '.join(log_metrics)
        logging.info(os.linesep + log_info + os.linesep)

    if args.verbose <= logging.DEBUG:
        if args.unlabel_test == 0:
            logging.debug(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
            logging.debug(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
        else:
            logging.debug(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor))
            logging.debug(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor))

    logging.info('# of params: %d' % model.total_parameters)
    logging.info(vars(origin_args))
    logging.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # result_dict = runner.run_some_tensors(model, data_processor.get_train_data(epoch=-1, model=model), data_processor,
    #                                       dict_keys=['sth'])
    # pickle.dump(result_dict, open('./sth.pk', 'rb'))
    return


if __name__ == '__main__':
    main()

    
    
    
    
