from flask import Flask, request
import pymongo, json, jieba, datetime, re
from pandas import DataFrame, Series
from dateutil.parser import parse

app = Flask(__name__)

mongo_client = pymongo.MongoClient('localhost')
scrapy_db = mongo_client['scrapy']
hype_col = scrapy_db['HypebeastItem']
cache_col = scrapy_db['cache']


@app.route('/key/get/<string:key>')
def key_get_str(key):
    print(key)
    args = request.args.to_dict()
    form = request.form

    print('get参数：' + json.dumps(args))
    option = args.get('option', None)
    period = args.get('period', None)
    cursor = Util.is_cached(key, period, option)
    if cursor is not None:
        print('return from cache')
        return json.dumps(cursor['data'])
    else:
        cursor = hype_col.find()
        data = Util.analyze(cursor, key, period)
        Util.cache(key, period, option, data)
        jv = json.dumps(data)
        return jv


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
    def analyze(cursor, key, period):
        m = {}
        for i in cursor:
            d = parse(i['datetime'])
            c = i['content']
            m[d] = m.get(d, 0) + len(re.findall(string=c, pattern=key))
        j = Series(m).resample(period).sort.to_json(orient='split')
        j = json.loads(j)
        j['index'] = [datetime.datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d') for x in j['index']]
        return j

    @staticmethod
    def is_cached(key, period, option):
        cursor = cache_col.find_one({'key': key, 'period': period, 'option': option}, {'_id': 0})
        return cursor

    @staticmethod
    def cache(key, period, option, data):
        d = dict()
        d['key'] = key
        d['period'] = period
        d['option'] = option
        d['data'] = data
        cache_col.insert_one(d)
        return True


if __name__ == '__main__':
    app.run(debug=True)
