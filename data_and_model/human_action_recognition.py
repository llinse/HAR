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
from wsgiref.simple_server import make_server
app = Flask(__name__)


data = []
for name in ["activity1.json","activity2.json","activity3.json","activity4.json"]:
    with open(name,'r', encoding = 'utf-8') as file:
        for line in file.readlines():
            dic = json.loads(line)
            data.append(dic)
df = DataFrame(data)
# df_title = df.columns.values.tolist()

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


Data1=[]
for item in ['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs']:
    dat1=[]
    for i in range(0,35):
        for j in range(1,7):
            dat = data1.loc[i,item]
            #dat = dat[j*64:(j+1)*64+64]
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
            #dat = dat[j*64:(j+1)*64+64]
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
            #dat = dat[j*64:(j+1)*64+64]
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
            #dat = dat[j*64:(j+1)*64+64]
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


order = 6
fs = 30.0  # sample rate, Hz
cutoff = 1.5
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)

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

Xtrain, Xtest, Ytrain, Ytest=train_test_split\
    (edata.loc[:,['accXs','accYs','accZs','gyrXs','gyrYs','gyrZs']],
     edata.loc[:,'activity'],test_size=0.1)
clf = model.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest)
print(score)

# # Actual class predictions
rf_predictions = model.predict(Xtest)
# Probabilities for each class
rf_probs = model.predict_proba(Xtest)


@app.route('/',methods=['post'])
def hello_world():
    accXs= request.values.get("accXs")
    accYs = request.values.get("accYs")
    accZs = request.values.get("accZs")
    gyrXs = request.values.get("gyrXs")
    gyrYs = request.values.get("gyrYs")
    gyrZs = request.values.get("gyrZs")

    accXs = json.loads(accXs)
    accYs = json.loads(accYs)
    accZs = json.loads(accZs)
    gyrXs = json.loads(gyrXs)
    gyrYs = json.loads(gyrYs)
    gyrZs = json.loads(gyrZs)

    df1=[accXs, accYs, accZs, gyrXs, gyrYs, gyrZs]
    Data = DataFrame(df1, index=['accXs', 'accYs', 'accZs', 'gyrXs', 'gyrYs', 'gyrZs'])
    Data = Data.stack()
    Data = Data[:-3]
    Data = Data.unstack(0)

    rf_predictions = model.predict(Data)
    n = len(rf_predictions)
    rf_predictions1 = rf_predictions[1:int(n / 6)]
    rf_predictions2 = rf_predictions[int(n / 6):int(n / 3)]
    rf_predictions3 = rf_predictions[int(n / 3):int(n / 2)]
    rf_predictions4 = rf_predictions[int(n / 2):int(n * 4 / 6)]
    rf_predictions5 = rf_predictions[int(n * 4 / 6):int(n * 5 / 6)]
    rf_predictions6 = rf_predictions[int(n * 5 / 6):-1]
    count_dict1 = {}
    for label in rf_predictions1:
        if label not in count_dict1.keys():
            count_dict1[label] = 0
        count_dict1[label] += 1
    max1 = max(zip(count_dict1.values(), count_dict1.keys()))[1]
    count_dict2 = {}
    for label in rf_predictions2:
        if label not in count_dict2.keys():
            count_dict2[label] = 0
        count_dict2[label] += 1
    max2 = max(zip(count_dict2.values(), count_dict2.keys()))[1]
    count_dict3 = {}
    for label in rf_predictions3:
        if label not in count_dict3.keys():
            count_dict3[label] = 0
        count_dict3[label] += 1
    max3 = max(zip(count_dict3.values(), count_dict3.keys()))[1]
    count_dict4 = {}
    for label in rf_predictions4:
        if label not in count_dict4.keys():
            count_dict4[label] = 0
        count_dict4[label] += 1
    max4 = max(zip(count_dict4.values(), count_dict4.keys()))[1]
    count_dict5 = {}
    for label in rf_predictions5:
        if label not in count_dict5.keys():
            count_dict5[label] = 0
        count_dict5[label] += 1
    max5 = max(zip(count_dict5.values(), count_dict5.keys()))[1]
    count_dict6 = {}
    for label in rf_predictions6:
        if label not in count_dict6.keys():
            count_dict6[label] = 0
        count_dict6[label] += 1
    max6 = max(zip(count_dict6.values(), count_dict6.keys()))[1]
    a = [max1, max2, max3, max4, max5, max6]
    for i in range(len(a) - 1, 0, -1):
        if a[i] == a[i - 1]:
            del a[i]
    b = [str(j) for j in a]
    resu = ''.join(b)
    return resu

if __name__ == '__main__':
    app.run(host='172.17.0.13',debug=True)
    # server = make_server('172.17.0.13',22,app)
    # server.serve_forever()
    # app.run()
