import akshare as ak

stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol="sh000016")
print(stock_zh_index_daily_df)

"""
date：日期，表示这些数据对应的交易日期。
open：开盘价，表示在该交易日开始时的股票价格。
high：最高价，表示在该交易日内股票的最高交易价格。
low：最低价，表示在该交易日内股票的最低交易价格。
close：收盘价，表示在该交易日结束时的股票价格。
volume：成交量，表示在该交易日内的股票交易数量。
https://zhuanlan.zhihu.com/p/115769021
https://akshare.akfamily.xyz/data/stock/stock.html#id2
"""
