import scipy as sp
from scipy import optimize
import numpy as np
from numpy import array
from scipy import linalg
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from dateutil.parser import parse
import pymongo, jieba, datetime, re, json
import pandas as pd


# tau=0.045
def fourier_func(x, *a):
    ret = a[0]
    w = a[1]
    start = 2
    if len(a) % 2 == 1:
        start = 1
    for deg in range(start, len(a), 2):
        ret += a[deg] * np.cos(deg * w * x) + a[deg + 1] * np.sin(deg * w * x)
    return ret


def analyze(content, key):
    m = {}
    for i in content:
        d = parse(i['datetime'])
        c = i['content']
        #         m[d] = m.get(d,0)+len(re.findall(string=c,pattern=key))
        have = 0
        if len(re.findall(string=c, pattern=key)) > 0:
            have = 1
        m[d] = m.get(d, 0) + have
    return Series(m)


def change_index(s1: Series, s3: Series, start=0, end=-1):
    '''
    s1:原始数据
    s3:拟合曲线
    start:s3拟合原始数据的开始
    end:s3拟合原始数据的结束
    '''
    start_date = s1.index[start]
    end_date = s1.index[end]
    length = len(s3.index)
    curve_date = pd.date_range(start=start_date, periods=length, freq=(end_date - start_date) / length)
    s3.index = curve_date


def curve(key_word='无', n=10, start=0, end=-1, freq='Q-JAN', curve_func=fourier_func, predict=True,
          col=pymongo.MongoClient('localhost')['scrapy']['predict'],
          src_col=pymongo.MongoClient('localhost')['scrapy']['HypebeastItem']):
    '''
    给出数据，进行拟合，并进行显示和保存
    key_word:拟合关键词
    n:傅里叶级数的阶数
    start:拟合数据的开始
    end:拟合数据的阶数
    freq:周期
    curve_func:用于拟合的函数
    predict:是否显示预测线
    col:用于保存的MongoDB
    src_col:原始数据
    '''
    cursor = src_col.find()
    out = analyze(cursor, key_word)
    sm = out.resample(freq).sum()
    smr = Series(data=sm.data, index=[x for x in range(0, len(sm.index))])
    x_data = smr.index[start:end]
    y_data = smr.data[start:end]
    func = fourier_func
    params, params_convariance = optimize.curve_fit(func, x_data, y_data, [1.0] * n)
    print(params)
    plt.plot(x_data, y_data, marker='o', color='g',
             label='data')

    plt.plot(x_data, func(x_data, *params), color='red',
             label='fit_curved')

    x_temp = np.linspace(x_data[0], x_data[-1] + 3, 100)

    if predict:
        plt.plot(x_temp, func(x_temp, *params), 'b--', label='real_fit_curved')

    plt.title(key_word)
    plt.legend()
    plt.show()
    #     保存数据
    s3 = Series(func(x_temp, *params), x_temp)

    change_index(sm, s3, start, end)

    df1 = DataFrame({'data': sm})
    df2 = DataFrame({'curve': s3})

    df3 = df1.add(df2, fill_value=0)

    df3.plot()

    j = json.loads(df3.to_json(orient='split', date_unit='s'))

    j['key'] = key_word
    j['freq'] = freq

    #     col.insert_one(j)
    col.update_one({"key": j['key'], "freq": j['freq']}, {"$set": j}, upsert=True)


curve(key_word='红', start=1)
