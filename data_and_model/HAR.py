import json
import numpy as np
from pandas import DataFrame
from scipy.signal import butter, lfilter, freqz, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from numpy import array
from numpy import dstack
from numpy import unique
from pandas import concat

from flask import Flask,request
#from wsgiref.simple_server import make_server
app = Flask(__name__)


data = []
for name in ["activity1.json","activity2.json","activity3.json","activity4.json"]:
    with open(name,'r', encoding = 'utf-8') as file:
        for line in file.readlines():
            dic = json.loads(line)
            data.append(dic)
df = DataFrame(data)
# df_title = df.columns.values.tolist()  # 获得列名

# # activity class balance
# def class_breakdown(data):
#     df = DataFrame(data)
#     counts = df.groupby('activity').size()
#     counts = counts.values
#     print(counts)
#     for i in range(len(counts)):
#         percent = counts[i] / len(df) * 100
#         print('Class=%d, total=%d, percentage=%.3f' % (i+1, counts[i], percent))
# # class_breakdown(data)


# 单独取出一个活动的数据
column_name=['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs','activity']
data_1= df.query('activity == "1"')
data1=data_1.loc[:,column_name]
data_2= df.query('activity == "2"')
data2=data_2.loc[:,column_name]
data_3= df.query('activity == "3"')
data3=data_3.loc[:,column_name]
data_4= df.query('activity == "4"')
data4=data_4.loc[:,column_name]
adata=df.loc[:,column_name]

# 数据处理与滤波
Data1=[]
for item in ['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs']:
    dat1=[]
    for i in range(0,35):
        for j in range(1,7):
            dat = data1.loc[i,item]
            #dat = dat[j*64:(j+1)*64+64]    #舍去开头结尾部分数据，每隔128个数据带有50%重叠取值
            dat = dat[j*128:(j+1)*128]
            dat1.extend(dat)
    Data1.append(dat1)
Data1 = DataFrame(Data1,index=['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs'])
Data1 = Data1.stack()
Data1 = Data1.unstack(0)
raw_dic1 = {}
for i in range(0, 200):
    dat2 = Data1.loc[i*128:i*128+127, :]
    dat2 = DataFrame(dat2)
    key = i+1
    raw_dic1[key] = dat2


Data2=[]
for item in ['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs']:
    dat1=[]
    for i in range(35,62):
        for j in range(1,7):
            dat = data2.loc[i,item]
            #dat = dat[j*64:(j+1)*64+64]    #舍去开头结尾部分数据，每隔128个数据带有50%重叠取值
            dat = dat[j*128:(j+1)*128]
            dat1.extend(dat)
    Data2.append(dat1)
Data2 = DataFrame(Data2,index=['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs'])
Data2 = Data2.stack()
Data2 = Data2.unstack(0)
raw_dic2 = {}
for i in range(0, 162):
    dat2 = Data2.loc[i*128:i*128+127, :]
    dat2 = DataFrame(dat2)
    key = i+1
    raw_dic2[key] = dat2


Data3=[]
for item in ['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs']:
    dat1=[]
    for i in range(63,90):
        for j in range(1,7):
            dat = data3.loc[i,item]
            #dat = dat[j*64:(j+1)*64+64]    #舍去开头结尾部分数据，每隔128个数据带有50%重叠取值
            dat = dat[j*128:(j+1)*128]
            dat1.extend(dat)
    Data3.append(dat1)
Data3 = DataFrame(Data3,index=['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs'])
Data3 = Data3.stack()
Data3 = Data3.unstack(0)
raw_dic3 = {}
for i in range(0, 162):
    dat2 = Data3.loc[i*128:i*128+127, :]
    dat2 = DataFrame(dat2)
    key = i+1
    raw_dic3[key] = dat2


Data4=[]
for item in ['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs']:
    dat1=[]
    for i in range(91,119):
        for j in range(1,7):
            dat = data4.loc[i,item]
            #dat = dat[j*64:(j+1)*64+64]    #舍去开头结尾部分数据，每隔128个数据带有50%重叠取值
            dat = dat[j*128:(j+1)*128]
            dat1.extend(dat)
    Data4.append(dat1)
Data4 = DataFrame(Data4,index=['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs'])
Data4 = Data4.stack()
Data4 = Data4.unstack(0)
raw_dic4 = {}
for i in range(0, 168):
    dat2 = Data4.loc[i*128:i*128+127, :]
    dat2 = DataFrame(dat2)
    key = i+1
    raw_dic4[key] = dat2


# #低通滤波
order = 6
fs = 30.0  # sample rate, Hz
cutoff = 1.5
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)
#对所有数据进行低通滤波
for i in range(1, 200):
     for j in ['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs']:
         signal = np.array(raw_dic1[i][j])
         low_signal = lfilter(b, a, signal)
         raw_dic1[i].loc[:, (j)] = np.array(low_signal)
for i in range(1, 162):
     for j in ['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs']:
         signal = np.array(raw_dic2[i][j])
         low_signal = lfilter(b, a, signal)
         raw_dic2[i].loc[:, (j)] = np.array(low_signal)
for i in range(1, 162):
     for j in ['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs']:
         signal = np.array(raw_dic3[i][j])
         low_signal = lfilter(b, a, signal)
         raw_dic3[i].loc[:, (j)] = np.array(low_signal)
for i in range(1, 168):
     for j in ['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs']:
         signal = np.array(raw_dic4[i][j])
         low_signal = lfilter(b, a, signal)
         raw_dic4[i].loc[:, (j)] = np.array(low_signal)

# #去除50%的重叠
# def to_series(windows):
# 	series = list()
# 	for window in windows:
# 		# remove the overlap from the window
# 		half = int(len(window) / 2) - 1
# 		for value in window[-half:]:
# 			series.append(value)
# 	return series

frame1=[]
for i in range(1,200):
    frame1.append(raw_dic1[i])
edata1=concat(frame1)

frame2=[]
for i in range(1,162):
    frame2.append(raw_dic2[i])
edata2=concat(frame2)

frame3=[]
for i in range(1,162):
    frame3.append(raw_dic3[i])
edata3=concat(frame3)

frame4=[]
for i in range(1,168):
    frame4.append(raw_dic4[i])
edata4=concat(frame4)


edata1['activity']=1
edata2['activity']=2
edata3['activity']=3
edata4['activity']=4

edata=concat([edata1,edata2,edata3,edata4],ignore_index=True) # 88064 rows x 7columns

# edat1=edata[0:100]
# ax=[]
# for i in range(1,100):
#     ax=ax.append(edat1[i,1])
# print(ax)
#
# print(edat1)

#for i in range(1,1000):


# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100,
                               bootstrap = True,
                               max_features = 'sqrt')

Xtrain, Xtest, Ytrain, Ytest=train_test_split(edata.loc[:,['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs']],edata.loc[:,'activity'],test_size=0.1)

clf = model.fit(Xtrain, Ytrain)                          #实例化训练集
score = clf.score(Xtest, Ytest)                        #返回预测的准确度
#print(score)


# # Actual class predictions
rf_predictions = model.predict(Xtest)
# Probabilities for each class
rf_probs = model.predict_proba(Xtest)


@app.route('/predict',methods=['post'])
def hello_world():
    accXs= request.values.get("accXs")
    accXs= json.loads(accXs)
    accYs = request.values.get("accYs")
    accYs = json.loads(accYs)
    accZs = request.values.get("accZs")
    accZs = json.loads(accZs)
    gyrXs = request.values.get("gyrXs")
    gyrXs = json.loads(gyrXs)
    gyrYs = request.values.get("gyrYs")
    gyrYs = json.loads(gyrYs)
    gyrZs = request.values.get("gyrZs")
    gyrZs = json.loads(gyrZs)
    ax=[]
    with open(accXs,'r', encoding = 'utf-8') as file:
        for line in file.readlines():
            dic = json.loads(line)
            ax.append(dic)
    ay = []
    with open(accYs, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            dic = json.loads(line)
            ay.append(dic)
    az = []
    with open(accZs, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            dic = json.loads(line)
            az.append(dic)
    gx = []
    with open(gyrXs, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            dic = json.loads(line)
            gx.append(dic)
    gy = []
    with open(gyrYs, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            dic = json.loads(line)
            gy.append(dic)
    gz = []
    with open(gyrZs, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            dic = json.loads(line)
            gz.append(dic)
    Data = DataFrame([ax,ay,az,gx,gy,gz], index=['accXs', 'accYs', 'accZs', 'gyrXs', 'gyrYs', 'gyrZs'])
    Data = Data.stack()
    Data = Data.unstack(0)
    rf_predictions = model.predict(Data)
    temp1 = 0
    for i in [1,2,3,4]:
        if rf_predictions[0:235].count(i) > temp1:
            max1 = i
            temp1 = rf_predictions[0:235].count(i)
    temp2 = 0
    for i in [1,2,3,4]:
        if rf_predictions[235:470].count(i) > temp2:
            max2 = i
            temp2 = rf_predictions[235:470].count(i)
    temp3 = 0
    for i in [1,2,3,4]:
        if rf_predictions[470:705].count(i) > temp3:
            max3 = i
            temp3 = rf_predictions[470:705].count(i)
    temp4 = 0
    for i in [1,2,3,4]:
        if rf_predictions[705:940].count(i) > temp4:
            max4 = i
            temp4 = rf_predictions[705:940].count(i)
    temp5 = 0
    for i in [1,2,3,4]:
        if rf_predictions[940:1175].count(i) > temp5:
            max5 = i
            temp5 = rf_predictions[940:1175].count(i)
    temp6 = 0
    for i in [1,2,3,4]:
        if rf_predictions[1175:1408].count(i) > temp6:
            max6 = i
            temp6 = rf_predictions[1175:1408].count(i)
    resu=[max1,max2,max3,max4,max5,max6]
    return resu

if __name__ == '__main__':
    app.run(host='172.17.0.13',debug=True,port=22)
