import pandas as pd
df = pd.read_csv('stock_data.csv')

#把日期转换为时间格式并设置成索引
columns = ['日期','开盘', '最高', '收盘', '最低', '成交量']
df['日期']=pd.to_datetime(df['日期'])
df = df[columns]
df.set_index('日期', inplace=True)

#异常值处理、缺失值处理、数据归一化
for col in columns:
    if col == '日期':
        continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    outlier_condition = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
    
df.interpolate(method='time')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scalers = {}

for col in df.columns:
    scaler = MinMaxScaler()
    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    scalers[col] = scaler
print(df)

#划分数据集
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, shuffle=False)

import numpy as np
def data_split(dataset, look_back=100):
    X, y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        y.append(dataset[i + look_back, :])
    return np.array(X), np.array(y)

X_train, y_train = data_split(train.values)
X_test, y_test = data_split(test.values)

#构建神经网络
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Dropout, Input
from keras.optimizers import Nadam

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu"
    ),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(5)
])
model.compile(optimizer=Nadam(learning_rate=0.001), loss="mse")

#训练模型
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

history = model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test),
    shuffle=False,
    callbacks=[es]
)
model.save('股票数据预测.h5')

def inverse_scaling(df, scalers):
    #归一化数据逆缩放
    df_inverse = pd.DataFrame()
    for col in df.columns:
        df_inverse[col] = scalers[col].inverse_transform(df[col].values.reshape(-1, 1)).flatten()
    return df_inverse

#反归一化函数
df_inverse = inverse_scaling(df, scalers)
df_inverse['日期'] = df.index
df_inverse.set_index('日期', inplace=True)
print(df_inverse)#原数据
df_inverse = df_inverse.rename(columns={'开盘': 'open', '最高': 'high', '收盘': 'close', '最低': 'low', '成交量': 'volume'})

def predict(X, y, scalers):
    # 预测
    y_pred = model.predict(X)
    y_pred = pd.DataFrame(y_pred, columns=y.columns)
    y_pred = inverse_scaling(y_pred, scalers)
    y_true = inverse_scaling(y, scalers)
    return y_pred, y_true

y_pred_train, y_true_train = predict(X_train, pd.DataFrame(y_train, columns=test.columns), scalers)
y_pred_test, y_true_test = predict(X_test, pd.DataFrame(y_test, columns=test.columns), scalers)

import matplotlib.pyplot as plt
# 创建一个新的图表
plt.figure()

# 计算偏移量
offset = len(y_true_train) - 1

# 绘制训练集的真实值和预测值
plt.plot(range(len(y_true_train)), y_true_train['收盘'], label='训练集真实值')
plt.plot(range(len(y_pred_train) - 1), y_pred_train['收盘'][1:], label='训练集预测值')

# 绘制测试集的真实值和预测值，x 值加上偏移量
plt.plot(range(offset, offset + len(y_true_test)), y_true_test['收盘'], label='测试集真实值')
plt.plot(range(offset, offset + len(y_pred_test) - 1), y_pred_test['收盘'][1:], label='测试集预测值')

# 添加图例
plt.legend()

# 显示图表
plt.show()

#预测未来数据
last_part = X_test[0, :, :]

columns = ['开盘', '最高', '收盘', '最低', '成交量']
last_part = pd.DataFrame(last_part, columns=columns)

last_part = np.expand_dims(last_part, axis=0)
pred = model.predict(last_part)#包含时间步长
pred = inverse_scaling(pd.DataFrame(pred, columns=columns), scalers)
print(pred)

def predict_data(last_part, days=10):
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
    future_df = pd.DataFrame(future, columns=columns)
    return future_df

# days = y_test.shape[0]
days = 10
future = predict_data(last_part)

future=inverse_scaling(future,scalers)
future['日期'] = df.index[-days:]
future.set_index('日期', inplace=True)
print(future)