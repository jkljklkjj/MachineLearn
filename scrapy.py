import akshare as ak

stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20170301", end_date='20240528', adjust="")
print(stock_zh_a_hist_df)

stock_zh_a_hist_df.to_csv('stock_data.csv', index=False)

"""
名称	类型	描述
日期	object	交易日
股票代码	object	不带市场标识的股票代码
开盘	float64	开盘价
收盘	float64	收盘价
最高	float64	最高价
最低	float64	最低价
成交量	int64	注意单位: 手
成交额	float64	注意单位: 元
振幅	float64	注意单位: %
涨跌幅	float64	注意单位: %
涨跌额	float64	注意单位: 元
换手率	float64	注意单位: %
https://zhuanlan.zhihu.com/p/115769021
https://akshare.akfamily.xyz/data/stock/stock.html#id2
"""
