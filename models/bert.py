# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.OCLI_train_path = dataset + '/data/OCLI_train.csv'  # 推理训练集
        self.OCEMOTION_train_path = dataset + '/data/OCEMOTION_train.csv'  # 情感训练集
        self.TNEWS_train_path = dataset + '/data/TNEWS_train.csv'  # 新闻训练集
        self.OCLI_test_path = dataset + '/data/OCNLI_a.csv'  # 推理测试集
        self.OCEMOTION_test_path = dataset + '/data/OCEMOTION_a.csv'  # 情感测试集
        self.TNEWS_test_path = dataset + '/data/TNEWS_a.csv'  # 新闻测试集

        self.OCLI_class_list = [0, 1, 2]
        self.OCEMOTION_class_list = ['sadness', 'like', 'happiness', 'fear', 'disgust', 'surprise', 'anger']
        self.TNEWS_class_list = [100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116]  # 类别名单

        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 200  # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 32  # mini-batch大小
        self.pad_size = 64  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain'
        # 读取预置的 Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.OCLI_submit_output_path = './submit/ocnli_predict.json'  # 提交结果输出路径
        self.OCEMOTION_submit_output_path = './submit/ocemotion_predict.json'  # 提交结果输出路径
        self.TNEWS_submit_output_path = './submit/tnews_predict.json'  # 提交结果输出路径
        self.cut_ratio = 0.9  # 90%作为训练集，10%作为验证集


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.OCNLI_fc = nn.Linear(config.hidden_size, 3)
        self.OCEMOTION_fc = nn.Linear(config.hidden_size, 7)
        self.TNEWS_fc = nn.Linear(config.hidden_size, 15)

    def forward(self, x, total_batch):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        if total_batch % 3 == 0:
            out = self.OCNLI_fc(pooled)
        elif total_batch % 3 == 1:
            out = self.OCEMOTION_fc(pooled)
        else:
            out = self.TNEWS_fc(pooled)
        return out
