import csv  # 用于读写 CSV 文件
import json  # 用于解析 JSON 数据
import requests  # 用于发送 HTTP 请求
from lxml import etree  # 用于解析 HTML 文档


# 定义 DataScraper 类，实现数据抓取功能
class DataScraper:
    # 初始化函数
    def __init__(self):
        # 定义一个字典，用于存储页面名称和页面英文名称的对应关系
        self.pagename_type = {
            "业绩报表": "RPT_LICO_FN_CPD",
            "业绩快报": "RPT_FCI_PERFORMANCEE",
            "业绩预告": "RPT_PUBLIC_OP_NEWPREDICT",
            "预约披露时间": "RPT_PUBLIC_BS_APPOIN",
            "资产负债表": "RPT_DMSK_FN_BALANCE",
            "利润表": "RPT_DMSK_FN_INCOME",
            "现金流量表": "RPT_DMSK_FN_CASHFLOW"
        }

        self.pagename_en = {
            "业绩报表": "yjbb",
            "业绩快报": "yjkb",
            "业绩预告": "yjyg",
            "预约披露时间": "yysj",
            "资产负债表": "zcfz",
            "利润表": "lrb",
            "现金流量表": "xjll"
        }

        # 定义一个列表，用于存储英文列名
        self.en_list = []

        # 定义一个抓取数据的 URL
        self.url = 'https://datacenter-web.eastmoney.com/api/data/v1/get'

        # 定义请求头
        self.headers = {
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Connection': 'closed',
            'Referer': 'https://data.eastmoney.com/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Google Chrome";v="111", "Not(A:Brand";v="8", "Chromium";v="111"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"'
        }

    # 定义一个函数，用于获取指定页面的表格数据
    def get_table(self, page):
        # 定义一个参数字典，用于传递抓取参数
        params = {
            'sortTypes': '-1,-1',  # 用于指定排序方式，这里设置为降序
            'reportName': self.table_type,  # 用于指定要抓取的报表类型
            'columns': 'ALL',  # 用于指定要抓取的列名
            'filter': f'(REPORT_DATE=\'{self.timePoint}\')'  # 用于指定查询条件，这里设置为查询指定日期范围的数据
        }

        # 如果要抓取的报表类型为 "RPT_LICO_FN_CPD"，则需要修改查询条件
        if self.table_type in ['RPT_LICO_FN_CPD']:
            params['filter'] = f'(REPORTDATE=\'{self.timePoint}\')'

        # 添加分页参数
        params['pageNumber'] = str(page)

        # 使用 requests 库发送 GET 请求，获取数据
        response = requests.get(url=self.url, params=params, headers=self.headers)

        # 使用 json 模块解析响应数据
        data = json.loads(response.text)

        # 如果数据抓取成功，则返回数据
        if data['result']:
            return data['result']['data']
        else:
            return

    # 定义一个函数，用于获取指定页面的表头信息
    def get_header(self, all_en_list):
        # 创建一个空列表，用于存储中文列名
        ch_list = []

        # 定义一个页面的 URL
        url = f'https://data.eastmoney.com/bbsj/{self.pagename_en[self.pagename]}.html'

        # 使用 requests 库发送 GET 请求，获取页面数据
        response = requests.get(url)

        # 使用 lxml 模块解析 HTML 文档
        res = etree.HTML(response.text)

        # 遍历所有英文列名，获取对应的中文列名
        for en in all_en_list:
            ch = ''.join(
                [i.strip() for i in res.xpath(f'//div[@class="dataview"]//table[1]//th[@data-field="{en}"]//text()')])

            # 如果找到了中文列名，则将其添加到列表中
            if ch:
                ch_list.append(ch)
                # 将英文列名添加到列表中
                self.en_list.append(en)

        # 返回中文列名列表
        return ch_list

    # 定义一个函数，用于将抓取的数据写到 CSV 文件中
    def write_header(self, table_data):
        # 打开一个 CSV 文件，用于写数据
        with open(self.filename, 'w', encoding='utf-8', newline='') as f:
            # 创建一个 CSV 写器
            writer = csv.writer(f)
            # 获取表头信息
            headers = self.get_header(list(table_data[0].keys()))

            # 使用 CSV 写器写表头
            writer.writerow(headers)

    # 定义一个函数，用于将抓取的数据写到 CSV 文件中
    def write_table(self, table_data):
        # 打开一个 CSV 文件，用于追加写数据
        with open(self.filename, 'a', encoding='utf-8', newline='') as csvfile:
            # 创建一个 CSV 写器
            writer = csv.writer(csvfile)
            # 遍历所有数据，并将其写到 CSV 文件中
            for item in table_data:
                # 创建一个空列表，用于存储一行的数据
                row = []
                # 遍历所有列名，获取对应的数据
                for key in item.keys():
                    # 如果列名在英文列名列表中，则将其添加到列表中
                    if key in self.en_list:
                        row.append(str(item[key]))

                # 使用 CSV 写器写一行数据
                print(row)
                writer.writerow(row)

    # 定义一个函数，用于获取时间列表
    def get_timeList(self):
        # 定义一个请求头，用于模拟浏览器请求
        headers = {
            'Referer': 'https://data.eastmoney.com/bbsj/202312.html'
        }
        # 使用 requests 库发送 GET 请求，获取页面数据
        response = requests.get('https://data.eastmoney.com/bbsj/202312.html', headers=headers)

        # 使用 lxml 模块解析 HTML 文档
        res = etree.HTML(response.text)

        # 使用xpath获取时间列表
        return res.xpath('//*[@id="filter_date"]//option/text()')

    # 定义一个运行函数，用于启动数据抓取程序
    def run(self):
        # 使用 get_timeList 函数获取时间列表
        self.timeList = self.get_timeList()
        # 遍历时间列表，打印时间
        for index, value in enumerate(self.timeList):
            if (index + 1) % 5 == 0:
                print(value)
            else:
                print(value, end=' ; ')

        # 输入要抓取的时间点
        self.timePoint = str(input('\n请选择时间（可选项如上）:'))
        # 输入要抓取的报表类型
        self.pagename = str(
            input('请输入报表类型（业绩报表;业绩快报;业绩预告;预约披露时间;资产负债表;利润表；现金流量表）:'))

        # 判断输入的时间点和报表类型是否正确
        assert self.timePoint in self.timeList, '时间输入错误'
        assert self.pagename in list(self.pagename_type.keys()), '报表类型输入错误'

        # 根据输入的报表类型获取对应的表格类型



        self.table_type = self.pagename_type[self.pagename]
        self.filename = f'{self.pagename}_{self.timePoint}.csv'

        self.write_header(self.get_table(1))
        page = 1
        while True:
            table = self.get_table(page)
            if table:
                self.write_table(table)

            else:
                break
            page += 1


if __name__ == '__main__':
    scraper = DataScraper()
    scraper.run()


