# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, OCNLI_train_iter, OCNLI_dev_iter, OCEMOTION_train_iter, OCEMOTION_dev_iter, TNEWS_train_iter,
          TNEWS_dev_iter):
    start_time = time.time()
    model.train()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # t_total: total number of training steps for the learning
    t_total = max([len(OCNLI_train_iter), len(OCEMOTION_train_iter), len(TNEWS_train_iter)])
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=t_total * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    dev_best_acc = float(0.0)
    # OCEMOTION_dev_best_loss = float('inf')
    # TNEWS_dev_best_loss = float('inf')
    first_task = []
    second_task = []
    third_task = []

    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        last_improve = total_batch
        for i in range(t_total):
            # 计算第一个loss
            trains, labels = next(OCNLI_train_iter, (None, None))
            outputs, loss1 = model(trains, labels, 0)

            # 第二个loss
            trains, labels = next(OCEMOTION_train_iter, (None, None))
            outputs, loss2 = model(trains, labels, 1)

            # 第三个loss
            trains, labels = next(TNEWS_train_iter, (None, None))
            outputs, loss3 = model(trains, labels, 2)

            loss = loss1 + loss2 + loss3

            # 把参数梯度设为0
            model.zero_grad()
            # 对于标量求导
            loss.backward()
            # 更新所有的参数
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                # 如果验证集的结果都不下降了，直接退出
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)

                OCNLI_dev_acc, OCNLI_dev_loss = evaluate(config, model, 0, OCNLI_dev_iter)
                first_task.append((OCNLI_dev_acc, OCNLI_dev_loss))
                OCEMOTION_dev_acc, OCEMOTION_dev_loss = evaluate(config, model, 1, OCEMOTION_dev_iter)
                second_task.append((OCEMOTION_dev_acc, OCEMOTION_dev_loss))
                TNEWS_dev_acc, TNEWS_dev_loss = evaluate(config, model, 2, TNEWS_dev_iter)
                third_task.append((TNEWS_dev_acc, TNEWS_dev_loss))

                dev_loss = sum([OCNLI_dev_loss, OCEMOTION_dev_loss, TNEWS_dev_loss]) / 3
                dev_acc = sum([OCNLI_dev_acc, OCEMOTION_dev_acc, TNEWS_dev_acc]) / 3
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    last_improve = total_batch
                    torch.save(model.state_dict(), config.save_path)

                time_dif = get_time_dif(start_time)
                # msg = "OCNLI_Iter: {0:>6}, Val Loss: {1:>5.2}, Val Acc: {2:>6.2%}"
                # print(msg.format(total_batch, OCNLI_dev_loss, OCNLI_dev_acc))
                # msg = "OCEMOTION_Iter: {0:>6}, Val Loss: {1:>5.2}, Val Acc: {2:>6.2%}"
                # print(msg.format(total_batch, OCEMOTION_dev_loss, OCEMOTION_dev_acc))
                # msg = "TNEWS_Iter: {0:>6}, Val Loss: {1:>5.2}, Val Acc: {2:>6.2%}"
                # print(msg.format(total_batch, TNEWS_dev_loss, TNEWS_dev_acc))
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  ' \
                      'Val Acc: {4:>6.2%},  Time: {5} '
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        # if flag:
        #     break
        for p in optimizer.param_groups:
            p['lr'] *= 0.9
            print("The learning rate is {}".format(p['lr']))
    write_file('./first_task.txt', first_task)
    write_file('./second_task.txt', second_task)
    write_file('./third_task.txt', third_task)


def write_file(file_name, task_data):
    with open(file_name, 'w') as fg:
        for acc, loss in task_data:
            fg.write(str(acc)+'\t' + str(loss) + '\n')


# def get_loss(labels, outputs, log_var, config):
#     precision = torch.exp(-log_var)
#     diff = F.cross_entropy(outputs, labels)
#     loss = torch.sum(precision * diff + log_var, -1).to(config.device)
#     return torch.mean(loss)


#     test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, total_batch, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    # 什么时候应该跳出来
    n = len(data_iter)
    with torch.no_grad():
        for i in range(n):
            texts, labels = next(data_iter)
            outputs, loss = model(texts, labels, total_batch)
            # loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
