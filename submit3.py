# coding: UTF-8
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from importlib import import_module
from utils import DatasetIterater

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def load_testdata(config, path, pad_size=32):
    contents = []
    data = pd.read_csv(path, names=['content'], sep='\t')
    for i in tqdm(range(len(data))):
        content = data.iloc[i]['content']
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
        contents.append((token_ids, 1, seq_len, mask))
    return contents


def submit_iterator(dataset, config, batch_size):
    iter = DatasetIterater(dataset, batch_size, config.device)
    return iter


def submit_evaluate(model, data_iter):
    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
    return predict_all


def submit_test(config, model, test_iter, output_path):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    predict_label = submit_evaluate(model, test_iter)
    final_result = []
    
    for i in range(len(predict_label)):
        dic = {}
        dic["id"] = str(i)
        dic["label"] = str(config.class_list[predict_label[i]])
        final_result.append(dic)
    # 输出json文件
    import json
    with open(output_path, 'w') as f:
        for each in final_result:
            json_str = json.dumps(each)  # dumps
            f.write(json_str)
            f.write('\n')


if __name__ == '__main__':
    dataset = '.'  # 数据集

    model_name = 'bert3'  # bert
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
    model = x.Model(config).to(config.device)

    test_data = load_testdata(config, config.test_path, pad_size=32)
    test_iter = submit_iterator(test_data, config, len(test_data))
    submit_test(config, model, test_iter, config.submit_output_path)