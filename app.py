from flask import Flask, request
import pymongo, json, jieba, datetime, re
from pandas import DataFrame, Series
from dateutil.parser import parse

app = Flask(__name__)

mongo_client = pymongo.MongoClient('localhost')
scrapy_db = mongo_client['scrapy']
hype_col = scrapy_db['HypebeastItem']
cache_col = scrapy_db['cache']
DIMENSION = 'dimension'
KEY = 'key'
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


if __name__ == '__main__':
    app.run(debug=True)
