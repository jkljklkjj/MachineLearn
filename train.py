import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# 假设你的数据在'close'列
data = stock_zh_index_daily_df["close"]

# 将数据转换为浮点类型
data = data.astype("float32")

# 创建ARIMA模型
model = ARIMA(data, order=(5, 1, 0))

# 拟合模型
model_fit = model.fit(disp=0)

# 进行预测
forecast, stderr, conf_int = model_fit.forecast(steps=1)

print("Forecast: %f" % forecast)
