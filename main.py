# coding: UTF-8
import time
import torch
import numpy as np
import pandas as pd
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

dataset = '.'  # 数据集

model_name = 'bert'  # bert
# model_name = args.model  # bert
# 动态导入模块
x = import_module('models.' + model_name)
# 配置参数
config = x.Config(dataset)

# 固定以下参数是为了保证每次结果一样
np.random.seed(1)
# 为CPU设置种子用于生成随机数
torch.manual_seed(1)
# #为所有GPU设置随机种子
torch.cuda.manual_seed_all(1)
# 这个参数为True, 每次返回的卷积算法将是确定的，即默认算法
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

start_time = time.time()
print("Loading data...")
OCNLI_train, OCNLI_dev, OCEMOTION_train, OCEMOTION_dev, TNEWS_train, TNEWS_dev = build_dataset(config, mode='train')
OCNLI_train_iter = build_iterator(OCNLI_train, config)
OCEMOTION_train_iter = build_iterator(OCEMOTION_train, config)
TNEWS_train_iter = build_iterator(TNEWS_train, config)
OCNLI_dev_iter = build_iterator(OCNLI_dev, config)
OCEMOTION_dev_iter = build_iterator(OCEMOTION_dev, config)
TNEWS_dev_iter = build_iterator(TNEWS_dev, config)

time_dif = get_time_dif(start_time)

# train
model = x.Model(config).to(config.device)
train(config, model, OCNLI_train_iter, OCNLI_dev_iter, OCEMOTION_train_iter, OCEMOTION_dev_iter, TNEWS_train_iter,
      TNEWS_dev_iter)
