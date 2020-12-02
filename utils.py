# coding: UTF-8
import torch
import pandas as pd
from tqdm import tqdm
import time
from datetime import timedelta


PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config, mode='train'):

    def load_data_train(path, dataType, pad_size=32):
        train_contents = []
        dev_contents = []
        if dataType == 'OCNLI':
            data = pd.read_csv('./data/OCNLI_train.csv', names=['content1', 'content2', 'label'], sep='\t')
            # 将两个句子取前pad_size组合起来
            data['content'] = data['content1'].apply(lambda x: x[:int(pad_size/2)]) + data['content2'].apply(lambda x: x[:int(pad_size/2)])
        else:
            data = pd.read_csv(path, names=['content', 'label'], sep='\t')
        cut_ratio = int(0.9 * len(data))    # 分出10%的数据用于验证
        for i in tqdm(range(len(data))):
            content = data.iloc[i]['content']
            if dataType == 'OCNLI':
                label = config.OCLI_class_list[data.iloc[i]['label']]
            elif dataType == 'OCEMOTION':
                label = config.OCEMOTION_class_list.index(data.iloc[i]['label'])
            else:
                label = config.TNEWS_class_list.index(data.iloc[i]['label'])

            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            # 找到每个字对应的下标
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            if i <= cut_ratio:
                train_contents.append((token_ids, int(label), seq_len, mask))
            else:
                dev_contents.append((token_ids, int(label), seq_len, mask))
        return train_contents, dev_contents

    def load_data_test(path, dataType, pad_size=32):
        contents = []
        if dataType == 'OCNLI':
            data = pd.read_csv(path, names=['content1', 'content2'], sep='\t')
            # 将两个句子取前pad_size组合起来
            data['content'] = data['content1'].apply(lambda x: x[:int(pad_size/2)]) + data['content2'].apply(lambda x: x[:int(pad_size/2)])
        else:
            data = pd.read_csv(path, names=['content'], sep='\t')
        for i in tqdm(range(len(data))):
            content = data.iloc[i]['content']
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            # 找到每个字对应的下标
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, 1, seq_len, mask))
        return contents

    if mode == 'train':
        OCNLI_train, OCNLI_dev = load_data_train(config.OCLI_train_path, 'OCNLI', config.pad_size)
        OCEMOTION_train, OCEMOTION_dev = load_data_train(config.OCEMOTION_train_path, 'OCEMOTION', config.pad_size)
        TNEWS_train, TNEWS_dev = load_data_train(config.TNEWS_train_path, 'TNEWS', config.pad_size)
        return OCNLI_train, OCNLI_dev, OCEMOTION_train, OCEMOTION_dev, TNEWS_train, TNEWS_dev
    elif mode == 'test':
        OCNLI_test = load_data_test(config.OCLI_test_path, 'OCNLI', config.pad_size)
        OCEMOTION_test = load_data_test(config.OCEMOTION_test_path, 'OCEMOTION', config.pad_size)
        TNEWS_test = load_data_test(config.TNEWS_test_path, 'TNEWS', config.pad_size)
        return OCNLI_test, OCEMOTION_test, TNEWS_test
    else:
        return None


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        # 有多少个batch
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            # 保证循环到底之后重新开始
            self.index = 0
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def submit_iterator(dataset, config, batch_size):
    iter = DatasetIterater(dataset, batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))