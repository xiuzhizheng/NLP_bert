# coding: UTF-8
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from importlib import import_module
from utils import DatasetIterater
from utils import build_dataset

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


def submit_evaluate(model, data_iter, task_type):
    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts, task_type)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
    return predict_all


def submit_test(config, model, test_iter, output_path, task_type):
    """
    type: 0 OCNLI
          1 OCEMOTION
          2 TNEWS
    """
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    predict_label = submit_evaluate(model, test_iter, task_type)
    final_result = []
    
    for i in range(len(predict_label)):
        if task_type % 3 == 0:
            dic = {
                "id": str(i),
                "label": str(config.OCLI_class_list[predict_label[i]])
            }
        elif task_type % 3 == 1:
            dic = {
                "id": str(i),
                "label": str(config.OCEMOTION_class_list[predict_label[i]])
            }
        else:
            dic = {
                "id": str(i),
                "label": str(config.TNEWS_class_list[predict_label[i]])
            }
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

    model_name = 'bert'  # bert
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

    OCNLI_test, OCEMOTION_test, TNEWS_test = build_dataset(config, mode='test')
    OCNLI_test_iter = submit_iterator(OCNLI_test, config, len(OCNLI_test))
    OCEMOTION_test_iter = submit_iterator(OCEMOTION_test, config, len(OCEMOTION_test))
    TNEWS_test_iter = submit_iterator(TNEWS_test, config, len(TNEWS_test))
    # 第一个任务的提交
    submit_test(config, model, OCNLI_test_iter, config.OCLI_submit_output_path, 0)
    submit_test(config, model, OCEMOTION_test_iter, config.OCEMOTION_submit_output_path, 1)
    submit_test(config, model, TNEWS_test_iter, config.TNEWS_submit_output_path, 2)

