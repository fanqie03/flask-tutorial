from flask import Flask, request
import pymongo, json, jieba, datetime, re, time
from pandas import DataFrame, Series
from dateutil.parser import parse
from elasticsearch import Elasticsearch

app = Flask(__name__)

mongo_client = pymongo.MongoClient('localhost')
scrapy_db = mongo_client['scrapy']
hype_col = scrapy_db['HypebeastItem']
cache_col = scrapy_db['cache']
predict_col = scrapy_db['predict']
temp_db = mongo_client['temp']
taobao1_col = temp_db['taobao1']
client = Elasticsearch()
DIMENSION = 'dimension'
KEY = 'key'
FREQ = 'freq'
OPTION = 'option'
PERIOD = 'period'
WEBSITE = 'website'
RESULT = 'result'
DATA = 'data'


@app.route('/info/get', methods=['post', 'get'])
def key_get_str():
    args = request.args.to_dict()
    form = request.form

    print('get参数：' + json.dumps(args))
    print('post参数：')
    print(form)
    key = form.get(KEY, None)
    website = json.loads(form.get(WEBSITE, []))
    option = form.get(OPTION, None)
    period = form.get(PERIOD, None)
    # 维度
    condition = {DIMENSION: {KEY: key, WEBSITE: website, PERIOD: period}, OPTION: option}
    cursor = Util.is_cached(condition)
    if cursor is not None:
        print('return from cache')
        return json.dumps(cursor['data'])
    else:
        data = Util.analyze(condition)
        Util.cache(condition, data)
        jv = json.dumps(data)
        return jv


@app.route('/predict/get', methods=['post', 'get'])
def get_predict():
    args = request.args.to_dict()
    form = request.form
    print('get参数：' + json.dumps(args))
    print('post参数：')
    print(form)

    key = args[KEY]
    freq = args[FREQ]

    cursor = predict_col.find_one({KEY: key, FREQ: freq}, {'_id': 0})
    if cursor is None:
        return ''
    Util.mark_today(cursor)
    cursor['index'] = [datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d') for x in cursor['index']]
    return json.dumps(cursor)


@app.route('/recommend/get', methods=['post', 'get'])
def recommend():
    # 已经分析出来的数据
    raw_data = predict_col.find({}, {'_id': 0})
    predict_data = [x for x in raw_data]
    # 用于商品推荐的数据
    raw_data = taobao1_col.find({}, {'_id': 0})
    raw_data = [x for x in raw_data]
    j = Util.recommend(predict_data, raw_data)
    return json.dumps(j)


@app.route('/test', methods=['post', 'get'])
def test():
    args = request.args.to_dict()
    form = request.form
    print('args:')
    print(args)
    print('form:')
    print(form)
    print(form['key'])
    print(json.loads(form['website']))
    return '1'


class Util:

    @staticmethod
    def counter(cursor):
        m = {}
        for i in cursor:
            dt = parse(i['datetime'])
            m[dt] = m.get(dt, 0) + 1

        j = Series(m).sort_index().to_json(orient='split')
        j = json.loads(j)
        j['index'] = [datetime.datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d') for x in j['index']]
        return j

    @staticmethod
    def analyze(condition):
        dimension = condition[DIMENSION]
        website = dimension[WEBSITE]
        key = dimension[KEY]
        option = condition[OPTION]
        period = dimension[PERIOD]
        print(period)
        m = {}
        for site in website:
            cursor = scrapy_db[site].find()
            for i in cursor:
                d = parse(i['datetime'])
                c = i['content']
                m[d] = m.get(d, 0) + len(re.findall(string=c, pattern=key))
            d = Series(m).resample(period).sum().sort_index().to_dict()
            m.clear()
            m['index'] = [x.strftime('%Y-%m-%d') for x in d.keys()]
            m['data'] = [x for x in d.values()]
            # j['index'] = [datetime.datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d') for x in j['index']]
        return m

    @staticmethod
    def is_cached(condition):
        cursor = cache_col.find_one({DIMENSION: condition[DIMENSION], OPTION: condition[OPTION]}, {'_id': 0})
        return cursor

    @staticmethod
    def cache(condition, data):
        condition[DATA] = data
        cache_col.insert_one(condition)
        return True

    @staticmethod
    def get_predict(key, freq):
        ans = predict_col.find_one({'key': key, 'freq': 'freq'})
        return ans

    @staticmethod
    def striftodate(x):
        '''
        将时间戳转为日期，时间戳要求为10位，13位需要除以1000再传进来
        :param x:
        :return:
        '''
        return datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d')

    @staticmethod
    def mark_today(d):
        '''
        找到离今天最近的时间，添加该坐标
        :param d: 带index的时间戳数组的字典
        :return:
        '''
        index = 0
        min_ = 0x7fffffff
        now = int(time.time())
        for i in range(len(d['index'])):
            t = abs(now - int(d['index'][i]))
            if min_ > t:
                index = i
                min_ = t
        today = Util.striftodate(d['index'][index])
        #
        if d.get('model') is None:
            d['today'] = [today, d['data'][index][0]]
        # `model`字段不为空
        else:
            d['today'] = [today, d['data'][index][1]]

    @staticmethod
    def recommend(predict_data, raw_data, size=50):
        hot = []
        for d in predict_data:
            index = 0
            min_ = 0x7fffffff
            now = int(time.time())
            for i in range(len(d['index'])):
                t = abs(now - int(d['index'][i]))
                if min_ > t:
                    index = i
                    min_ = t
            if d.get('model') is None:
                next_data = d['data'][index + 1][0]
                today_data = d['data'][index][0]
                # `model`字段不为空
            else:
                next_data = d['data'][index + 1][1]
                today_data = d['data'][index][1]
            if (next_data > today_data):
                hot.append(d['key'])

        title = ''.join(hot)

        print(title)

        # 使用elasticsearch进行搜索
        response = client.search(index='taobao1', doc_type='_doc', body={
            "query": {
                "match": {
                    "title": title
                }
            },
            "highlight": {
                "fields": {
                    "title": {}
                }
            },
            "size": size
        })
        comment_items = response['hits']['hits']

        return comment_items


if __name__ == '__main__':
    # $ flask run
    app.run(debug=True)
