# coding: UTF-8
import torch
import numpy as np
from importlib import import_module
from utils import build_dataset, build_iterator


def submit_evaluate(model, data_iter, task_type):
    model.eval()
    predict_all = np.array([], dtype=int)
    n = len(data_iter)
    with torch.no_grad():
        for i in range(n):
            texts, labels = next(data_iter)
            outputs, _ = model(texts, labels, task_type)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
    print(len(predict_all))
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

    model = x.Model(config).to(config.device)

    OCNLI_test, OCEMOTION_test, TNEWS_test = build_dataset(config, mode='test')
    OCNLI_test_iter = build_iterator(OCNLI_test, config)
    OCEMOTION_test_iter = build_iterator(OCEMOTION_test, config)
    TNEWS_test_iter = build_iterator(TNEWS_test, config)
    # 第一个任务的提交
    submit_test(config, model, OCNLI_test_iter, config.OCLI_submit_output_path, 0)
    submit_test(config, model, OCEMOTION_test_iter, config.OCEMOTION_submit_output_path, 1)
    submit_test(config, model, TNEWS_test_iter, config.TNEWS_submit_output_path, 2)

