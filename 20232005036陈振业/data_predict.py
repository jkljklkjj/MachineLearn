from keras.models import load_model
import pandas as pd
import numpy as np
import data_preprocess
from keras.losses import MeanAbsoluteError as mse
from keras.optimizers import Nadam

def varify_predict(X, y, scalers):
    '''
    输入: 要预测的数据, 真实的数据, 归一化器
    输出: 预测值和真实值
    '''
    custom_objects = {'mse': mse, 'Nadam': Nadam}
    model = load_model('股票数据预测.h5', custom_objects=custom_objects)
    print('模型加载成功')
    print(model)
    y_pred = model.predict(X)
    y_pred = pd.DataFrame(y_pred, columns=y.columns)
    y_pred = data_preprocess.inverse_scaling(y_pred, scalers)
    y_true = data_preprocess.inverse_scaling(y, scalers)
    return y_pred, y_true

def predict_data(model,last_part, days=10):
    '''
    输入: 最新的一个数据, 预测天数
    输出: 未来天数的数据
    '''
    #已知数据(样本数，时间步长，特征数)
    future = []
    for i in range(days):#预测后面的天数
        next_day = model.predict(last_part,verbose=0)
        future.append(next_day)
        #所以把该变量和last_part合并，并删除last_part的第一个时间步长
        #lastpart为(样本数，时间步长，特征数)
        #next_day为(样本数，特征数)
        next_day = np.expand_dims(next_day, axis=1)
        last_part = np.concatenate((last_part, next_day), axis=1)
        last_part = last_part[:, -(last_part.shape[1] - 1):, :]
        
    future = np.squeeze(future, axis=1)
    future_df = pd.DataFrame(future, columns=['开盘', '最高', '收盘', '最低', '成交量'])
    return future_df