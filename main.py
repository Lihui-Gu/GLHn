import qlib
# regiodatetimeG_CN, REG_US]
from qlib.config import REG_US, REG_CN
# provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import datetime
from dataloader import DataLoader
from model import GLHn
import pandas as pd
from tqdm import tqdm
import torch
# 引入torch.nn并指定别名
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def mse(pred, label):
    loss = (pred - label) ** 2
    return torch.mean(loss)

def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    return mse(pred[mask], label[mask])

def pprint(*args):
    time = '['+str(datetime.datetime.utcnow()+
                    datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)
    with open('GLHn_log.txt', 'a') as f:
        print(time, *args, flush=True, file=f)


def get_matrix():
    matrix = np.load('./data/industry_concept_matrix.npy', allow_pickle=True)
    temp = np.zeros((matrix.shape[0], matrix.shape[0]))
    row = 0
    sum = 0
    while(row < temp.shape[0]):
        col = 0
        while(col < temp.shape[1]):
            value = (matrix[row] * matrix[col]).sum()
            temp[col][row] = value
            temp[row][col] = value
            col = col + 1
        row = row + 1
    row = 0
    new_temp = np.zeros((temp.shape[0], temp.shape[0]))
    while(row < temp.shape[0]):
        col = 0
        while(col < temp.shape[0]):
            new_temp[row][col] = temp[row][col] * temp[row][col] / (temp[row][row] * temp[col][col])
            col = col + 1
        row = row + 1
        return new_temp


def metric_fn(preds):
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='score', ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level =0).drop('datetime', axis = 1)
        
    for k in [1, 3, 5, 10, 20, 30, 50]:
        # 有多少类判别正确
        precision[k] = temp.groupby(level='datetime').apply(lambda x:((x.score[:k] * x.label[:k])>0).sum()/k).mean()
        # 预测为涨的有多少是真的涨
        recall[k] = temp.groupby(level='datetime').apply(lambda x:((x.score[:k]>0) * (x.label[:k]>0)).sum()/(x.score[:k]>0).sum()).mean()

    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score)).mean()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()
    return precision, recall, ic, rank_ic

def train_epoch(net, train_loader):
    new_temp = get_matrix()
    net.train()
    optimizer = optim.Adam(net.parameters())
    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        feature, label, stock_index, _ = train_loader.get(slc)
        feature = torch.tensor(feature, dtype = torch.float)
        label = torch.tensor(label, dtype = torch.float)
        batch_concept_matrix = new_temp[stock_index.values]
        batch_concept_matrix = batch_concept_matrix[:,stock_index.values]
        batch_concept_matrix = torch.tensor(batch_concept_matrix, dtype = torch.float)
        batch_concept_matrix = batch_concept_matrix.reshape(1, 1, len(batch_concept_matrix), -1)
        pred = net(feature, batch_concept_matrix)
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), 3.)
        optimizer.step()


def test_epoch(net, test_loader):
    new_temp = get_matrix()
    net.eval()
    losses = []
    preds = []
    for i, slc in tqdm(test_loader.iter_daily(), total=test_loader.daily_length):
        feature, label, stock_index, index = test_loader.get(slc)
        with torch.no_grad():
            feature = torch.tensor(feature, dtype = torch.float)
            label = torch.tensor(label, dtype = torch.float)
            batch_concept_matrix = new_temp[stock_index.values]
            batch_concept_matrix = batch_concept_matrix[:,stock_index.values]
            batch_concept_matrix = torch.tensor(batch_concept_matrix, dtype = torch.float)
            # print(feature)
            batch_concept_matrix = batch_concept_matrix.reshape(1, 1, len(batch_concept_matrix), -1)
            pred = net(feature, batch_concept_matrix)
            loss = loss_fn(pred, label)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))
            losses.append(loss.item())
    #evaluate
    preds = pd.concat(preds, axis=0)
    precision, recall, ic, rank_ic = metric_fn(preds)
    scores = ic
    return np.mean(losses), scores, precision, recall, ic, rank_ic

def main():
    train_start_date = '2007-01-01'
    train_end_date = '2014-12-31'
    valid_start_date = '2015-01-01'
    valid_end_date = '2016-12-31'
    test_start_date = '2017-01-01'
    test_end_date = '2020-12-31'

    start_time = datetime.datetime.strptime(train_start_date, '%Y-%m-%d')
    end_time = datetime.datetime.strptime(test_end_date, '%Y-%m-%d')
    train_end_time = datetime.datetime.strptime(train_end_date, '%Y-%m-%d')
    hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler', 'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time, 'fit_end_time': train_end_time, 'instruments': 'csi300', 'infer_processors': [{'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}}, {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}], 'learn_processors': [{'class': 'DropnaLabel'}], 'label': ['Ref($close, -1) / $close - 1']}}
    segments =  { 'train': (train_start_date, train_end_date), 'valid': (valid_start_date, valid_end_date), 'test': (test_start_date, test_end_date)}
    dataset = DatasetH(hanlder,segments)
    df_train, df_valid, df_test = dataset.prepare( ["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,)

    stock_index = np.load('stock_index.npy', allow_pickle=True).item()
    df_train['stock_index'] = df_train.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    train_loader = DataLoader(df_train["feature"], df_train["label"], df_train['stock_index'], device)
    df_valid['stock_index'] = df_valid.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    valid_loader = DataLoader(df_valid["feature"], df_valid["label"], df_valid['stock_index'], device)
    df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    test_loader = DataLoader(df_test["feature"], df_test["label"], df_test['stock_index'], device)
    net = GLHn()
    net.to(device)
    num_epoch = 200
    for epoch in range(num_epoch):
        pprint('Running', 'GLH','Epoch:', epoch)
        pprint('training...')
        train_epoch(net, train_loader)
        pprint('evaluating...')
        train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(net, train_loader)
        val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(net, valid_loader)
        test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(net, test_loader)
        pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
        pprint('train_score %.6f, valid_score %.6f, test_score %.6f'%(train_score, val_score, test_score))
        pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f'%(train_ic, val_ic, test_ic))
        pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f'%(train_rank_ic, val_rank_ic, test_rank_ic))
        pprint('Train Precision: ', train_precision)
        pprint('Valid Precision: ', val_precision)
        pprint('Test Precision: ', test_precision)
        pprint('Train Recall: ', train_recall)
        pprint('Valid Recall: ', val_recall)
        pprint('Test Recall: ', test_recall)


if __name__ == '__main__':
    main()
    
