'''
具有的功能：
1. 通过akshare库接收股票代码，获取股票数据，可以下载股票数据
2. 或者可以接收用户上传的股票数据
3. 对股票数据进行预处理
4. 根据已有的数据训练模型
5. 对未来10天的股票数据进行预测，并返回
6. 让用户选择是否可视化训练过程
'''

from flask import Flask, jsonify, request, session, send_file
import data_preprocess
import data_predict
import model_train
from keras.models import load_model
from keras.losses import MeanSquaredError as mse
from datetime import datetime, timedelta
from pandas import Timestamp
import pandas as pd
import akshare as ak
import secrets
import json

app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(16)

@app.route('/stock_get', methods=['POST'])
def stock_get():
    '''
    输入: 股票代码
    输出: 股票数据
    '''
    stock_code = request.form['stock_code']
    start = request.form.get('start', '20170301')
    end = request.form.get('end', '20240528')
    try:
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start, end_date=end, adjust="")
    except KeyError:
        return f'无法获取股票代码 {stock_code} 的数据'
    
    global stock_data
    stock_data = pd.DataFrame(df)
    stock_data['日期'] = pd.to_datetime(stock_data['日期']).dt.strftime('%Y-%m-%d')
    stock_data.set_index('日期', inplace=True)
    session['stock_code'] = stock_code

    data = stock_data.reset_index().to_dict(orient='records')
    stock_data.index = pd.to_datetime(stock_data.index)
    return json.dumps(data, ensure_ascii=False)

@app.route('/stock_upload', methods=['POST'])
def stock_upload():
    '''
    输入: 上传的股票数据
    要求: 文件格式为csv，至少包含日期、开盘、最高、收盘、最低、成交量
    输出: 股票数据
    '''
    file = request.files['file']
    global stock_data
    stock_data = pd.read_csv(file)
    columnns = ['日期', '开盘', '最高', '收盘', '最低', '成交量']
    for col in columnns:
        if col not in stock_data.columns:
            return '上传的文件格式不正确！'
    stock_data['日期'] = pd.to_datetime(stock_data['日期'])
    stock_data.set_index('日期', inplace=True)
    
    return '上传成功！'
    
@app.route('/stock_download', methods=['POST'])
def stock_download():
    '''
    输入: 股票代码
    输出: 下载股票数据
    '''
    #默认下载过去一年的数据
    end = request.form.get('end', (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'))
    start = request.form.get('start', (datetime.strptime(end, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d'))
    choice = request.form.get('choice', 1)
    
    print(type(stock_data.index))
    df = stock_data[start:end]
    if choice == 1:
        df = stock_preprocess(df)#预处理
    df.to_csv('stock_data.csv', index=False)
    return send_file('stock_data.csv', mimetype='text/csv', as_attachment=True)
    
def stock_preprocess(df):
    '''
    输入: 股票数据
    输出: 预处理后的股票数据
    '''
    df = data_preprocess.outlier_handle(df)
    df = data_preprocess.imputate_data(df)
    return df

@app.route('/stock_train', methods=['POST'])
def stock_train():
    '''
    输入: 预处理后的股票数据
    '''
    global stock_data
    global scalers
    global look_back
    
    columns = ['开盘', '最高', '收盘', '最低', '成交量']
    stock_data = stock_data[columns]
    stock_data = data_preprocess.outlier_handle(stock_data)
    stock_data = data_preprocess.imputate_data(stock_data)
    df, scalers = data_preprocess.normallize_data(stock_data)
    
    look_back = request.form.get('lookback', 80)
    train, test = data_preprocess.train_test_partition(df)
    X_train, y_train = data_preprocess.data_split(train.values, look_back)
    X_test, y_test = data_preprocess.data_split(test.values, look_back)
    
    history = model_train.model_train(X_train, y_train, X_test, y_test)
    last_val_loss = history.history['val_loss'][-1]
    print(last_val_loss)
    
    if last_val_loss > 0.005:
        text = '无法使用，请重新训练！'
    elif last_val_loss > 0.0045:
        text = '训练效果一般！'
    elif last_val_loss > 0.0040:
        text = '训练效果良好！'
    elif last_val_loss > 0.0038:
        text = '训练效果优秀！！'
    else :
        text = '训练效果极佳！！！'
    return '训练完成！模型'+text

@app.route('/model_upload', methods=['POST'])
def model_upload():
    '''
    输入: 上传的模型
    输出: 保存上传的模型
    '''
    global stock_data
    global scalers
    global look_back
    
    file = request.files['file']
    
    df,scalers = data_preprocess.normallize_data(stock_data)
    look_back = request.form.get('lookback', 80)
    
    file.save('股票数据预测.h5')
    return '上传成功！'

@app.route('/model_download', methods=['POST'])
def model_download():
    '''
    输入: 训练好的模型
    输出: 下载训练好的模型
    '''
    return send_file('股票数据预测.h5', mimetype='application/x-hdf', as_attachment=True)

@app.route('/stock_predict', methods=['POST'])
def stock_predict():
    '''
    输入: 预处理后的股票数据
    输出: 未来10天的股票数据
    '''
    global stock_data
    global scalers
    global look_back
    
    df = stock_data
    days = request.form.get('days', 10)
    model = load_model('股票数据预测.h5', custom_objects={'mse': mse})
    model.compile(loss=mse, optimizer='Nadam')
    
    last_part = df.tail(look_back).values
    last_part = pd.DataFrame(last_part, columns=df.columns)
    last_part = last_part.values.reshape(1, last_part.shape[0], last_part.shape[1])
    future = data_predict.predict_data(model, last_part, days=days)
    future = data_preprocess.inverse_scaling(future, scalers)
    print(future)
    
    return jsonify(future.to_dict())

if __name__ == '__main__':
    app.run(debug=True)