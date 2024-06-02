import matplotlib.pyplot as plt
import mplfinance as mpf

def kline(df,start_time,end_time):
    '''
    输入: 股票数据，开始时间，结束时间
    输出: K线图展示
    '''
    data = df.rename(columns={'开盘': 'open', '最高': 'high', '收盘': 'close', '最低': 'low', '成交量': 'volume'})
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 处理数据
    if start_time not in df.index or end_time not in df.index:
        raise Warning("start_time or end_time is out of index range!")
    data = data[start_time:end_time]

    # 添加图表
    global fig
    fig = plt.figure(figsize=(8, 4),dpi=200)
    ax = fig.add_axes([0,0.2,1,0.5])
    ax2 = fig.add_axes([0,0,1,0.2])
    # 绘制K线图
    mpf.plot(data,type='candle', style='charles', ax=ax, volume=ax2,mav=(10,5),mavcolors=('r','b'))

    ax.set_title("股票K线图")
    ax.grid(True)
    plt.show()
    
def history_visual(history):
    '''
    输入: 训练模型的历史记录
    输出: 训练损失和验证损失的可视化
    '''
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 绘制训练损失和验证损失
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.show()
    
def verify_predict(y_pred, y_true):
    '''
    输入: 预测值和真实值
    输出: 预测值和真实值的对比图
    '''
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))
    plt.plot(y_pred, label='预测值')
    plt.plot(y_true, label='真实值')
    plt.xlabel('日期')
    plt.ylabel('收盘价')
    plt.legend()
    plt.show()

def plot_predictions(y_true_train, y_pred_train, y_true_test, y_pred_test):
    plt.figure()

    offset = len(y_true_train) - 1
    plt.plot(range(len(y_true_train)), y_true_train['收盘'], label='训练集真实值')
    plt.plot(range(len(y_pred_train) - 1), y_pred_train['收盘'][1:], label='训练集预测值')

    # 绘制测试集的真实值和预测值，x 值加上偏移量
    plt.plot(range(offset, offset + len(y_true_test)), y_true_test['收盘'], label='测试集真实值')
    plt.plot(range(offset, offset + len(y_pred_test) - 1), y_pred_test['收盘'][1:], label='测试集预测值')

    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()