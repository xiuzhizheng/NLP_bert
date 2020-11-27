# coding: UTF-8
import torch
import pandas as pd
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):

    def load_dataset(path, pad_size=32):

        data = pd.read_csv('./data/OCNLI_train.csv', names=['content1', 'content2','label'], sep='\t')
        
        def get_contents(data,item='content'):
            contents = []
            for i in tqdm(range(len(data))):
                content,label = data.iloc[i][item], config.class_list.index(data.iloc[i]['label'])
                # todo
                # 为什么都是按单词分割
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
                contents.append((token_ids, int(label), seq_len, mask))
            return contents
        
        contents1 = get_contents(data,item='content1')
        contents2 = get_contents(data,item='content2')
        contents = []
        '''
        for i in range(len(contents1)):
            token_ids = []
            mask = []
            for j in range(len(contents1[i][0])):
                token_ids.append(abs(contents1[i][0][j]-contents2[i][0][j]))
                mask.append(0 if contents1[i][0][j]-contents2[i][0][j]==0 else 1)
            label = contents1[i][1]
            seq_len = contents1[i][2]
            contents.append((token_ids, int(label), seq_len, mask))
        '''
        for i in range(len(contents1)):
            token_ids = contents1[i][0]+contents2[i][0]
            mask = contents1[i][3]+contents2[i][3]
            label = contents1[i][1]
            seq_len = contents1[i][2]*2
            contents.append((token_ids, int(label),seq_len,mask))

        return contents
    
    dataset = load_dataset(config.train_path, config.pad_size)
    cut_ratio = int(0.8 * len(dataset))
    train = dataset[:cut_ratio]
    dev = dataset[cut_ratio+1:]
    return train, dev


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
            self.index = 0
            raise StopIteration
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


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
