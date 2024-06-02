import pandas as pd
import numpy as np

def read_data(filename):
    '''
    输入: 含有股票数据的csv文件
    输出: DataFrame格式的股票数据
    '''
    df = pd.read_csv(filename)
    columns = ['日期','开盘', '最高', '收盘', '最低', '成交量']
    df['日期']=pd.to_datetime(df['日期'])
    df = df[columns]
    df.set_index('日期', inplace=True)
    # print(df)
    return df

def normallize_data(df):
    '''
    输入: DataFrame格式的股票数据
    输出: 归一化后的股票数据和归一化器
    '''
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scalers = {}

    for col in df.columns:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        scalers[col] = scaler
    return df, scalers

def outlier_handle(df):
    '''
    输入: DataFrame格式的股票数据
    输出: 处理异常值后的股票数据
    '''
    for col in df.columns:
        if col == '日期':
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
        outlier_condition = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        # print(outlier_condition)
        return df

def imputate_data(df):
    '''
    输入: DataFrame格式的股票数据
    输出: 插值处理后的股票数据
    '''
    df.interpolate(method='time')
    return df

def train_test_partition(df, test_size=0.2, shuffle=False):
    #划分数据集
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=test_size, shuffle=shuffle)
    return train, test

def data_split(dataset, look_back=80):
    '''
    输入: DataFrame格式的股票数据
    输出: 包含时间步长的X和原来的y
    '''
    X, y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        y.append(dataset[i + look_back, :])
    return np.array(X), np.array(y)



def inverse_scaling(df, scalers):
    '''
    输入: 归一化后的股票数据和归一化器
    输出: 反归一化后的股票数据
    '''
    df_inverse = pd.DataFrame()
    for col in df.columns:
        df_inverse[col] = scalers[col].inverse_transform(df[col].values.reshape(-1, 1)).flatten()
    return df_inverse

def data_preprocess(filename):
    '''
    输入: 含有股票数据的csv文件
    输出: 处理后的股票数据
    '''
    df = read_data(filename)
    df = outlier_handle(df)
    df = imputate_data(df)
    df, scalers = normallize_data(df)
    return df, scalers

# data_preprocessing.py：包含数据预处理的代码，如读取数据、处理异常值、数据归一化等。

# model_building.py：包含构建神经网络模型的代码。

# model_training.py：包含训练模型的代码。

# model_prediction.py：包含预测和绘图的代码。

# main.py：调用以上四个文件的函数，并执行主程序。