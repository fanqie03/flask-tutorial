import tensorflow as tf
import random
import pymongo
from pandas import DataFrame,Series
import pandas as pd

# 正则化损失函数系数
REGULARIZATION_RATE = 0.05
# 隐藏层神经元数目
HIDDEN_LAYER=50
# 输出节点数目
OUTPUT_NODE=1
FREQ='10d'
KEYWORD='红'

train_percent = 0.8


def decompose_num(nums):
    '''
    切割数字
    如：nums=['2015', '07', '31']
    返回[2, 0, 1, 5, 0, 7, 3, 1]
    '''
    a=[]
    for num in nums:
        for j in num:
            a.append(int(j))
    return a
decompose_num(['2015', '07', '31'])


# 获取数据
col = pymongo.MongoClient('localhost')['scrapy']['HypebeastItem']
cursor = col.find()
s = analyze(cursor,KEYWORD)
s = s.resample(FREQ).sum()
# 去掉第一个和最后一个数据
ss = s[1:-1]

train_size = int(train_percent*len(ss))
test_size = len(ss)-train_size
train_data = ss[:train_size]
test_data = ss[train_size-1:]
raw_data = s
test_x = [decompose_num(str(t.date()).split('-')) for t in test_data.index]
test_y = [[t] for t in test_data]

resample_data=s
resample_split_data=ss

X=[decompose_num(str(t.date()).split('-')) for t in train_data.index]
Y = [[t] for t in train_data]

input_node=len(X[0])
print(input_node)



# 清空图
tf.reset_default_graph()

# 模型
# 权重
w1 = tf.Variable(tf.random_normal([input_node,HIDDEN_LAYER]))
w2 = tf.Variable(tf.random_normal([HIDDEN_LAYER,HIDDEN_LAYER]))
w3 = tf.Variable(tf.random_normal([HIDDEN_LAYER,OUTPUT_NODE]))
# 偏置值
biases1 = tf.Variable(tf.constant(0.1,shape=[HIDDEN_LAYER]))
biases2 = tf.Variable(tf.constant(0.1,shape=[HIDDEN_LAYER]))
biases3 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
# 输入输出占位符
x = tf.placeholder(tf.float32,shape=(None,input_node),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')
# 计算设置
a1 = tf.nn.sigmoid(tf.matmul(x,w1) + biases1)
a2 = tf.nn.sigmoid(tf.matmul(a1,w2) + biases2)
y = tf.nn.relu(tf.matmul(a2,w3) + biases3)

# a = tf.matmul(x,w1) + biases1
# y = tf.matmul(a,w2) + biases2

# 计算L2正则化损失函数
regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
# 计算模型正则化损失。一般只计算权重，而不用偏置值
regularization = regularizer(w1)+regularizer(w2)

# 定义mse
mse = tf.reduce_mean(tf.square(y_ - y))

# 定义损失函数
# y_为标准答案
# loss = mse
loss = mse + regularization
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


# 模型训练结果
model_x = [decompose_num(str(x.date()).split('-')) for x in pd.date_range(raw_data.index[0],periods=int(len(raw_data)*1.5),freq=FREQ)]
model_y = None

# 声明tf.train.Saver用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    STEPS = 10001
    for i in range(STEPS):
        if i % 1000 == 0:
            print("After %d training step(s), MSE on all data is %g" % (i,sess.run(loss,feed_dict={x:X,y_:Y})))
        sess.run(train_step,feed_dict={x:X,y_:Y})
    #   模型训练结果
    test_feed={x:model_x}
    model_y = sess.run(y,feed_dict=test_feed)
    model_y_s=Series([x[0] for x in model_y])
    out_data = model_y_s

    # 模型在测试集的表现
    print("In the test data, the model MSE is %g" % sess.run(loss,feed_dict={x:test_x,y_:test_y}))

    # 模型导出
    #     writer = tf.summary.FileWriter('log',sess.graph)
    # 模型持久化
    #     saver.save(sess,"model/{}.ckpt".format(KEYWORD))

    # 保存结果
    # 切换换回原来的坐标轴
    # 需要train_data,test_data,out_data
    out_data.index = pd.date_range(start=raw_data.index[0],freq=FREQ,periods=len(out_data))
    # 合并原始数据转成字典
    df_data1 = DataFrame({'curve':out_data})
    df_data2 = DataFrame({'data':train_data})
    df_data3 = DataFrame({'test':test_data})
    df_data = df_data1.add(df_data2,fill_value=0)
    df_data = df_data.add(df_data3,fill_value=0)
    df_data.plot(figsize=(20, 8.5))
#     j_data = json.loads(df_data.to_json(orient='split',date_unit='s'))
# 添加key，freq，option
#     j_data['key'] = KEYWORD
#     j_data['freq']=FREQ
#     j_data['option']='TensorFlow'
#     print(j_data)
# 保存到MongoDB
#     predict_col = pymongo.MongoClient('localhost')['scrapy']['predict']
#     predict_col.update_one({"key":j_data['key'],"freq":j_data['freq']},{"$set":j_data},upsert=True)

