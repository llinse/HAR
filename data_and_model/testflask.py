# from flask import Flask
# import sys
# app = Flask(__name__)
#
#
# @app.route('/')
# def Hello_word():
#   return "Hello word!"
#
# if __name__ == '__main__':
#     app.run(host='172.17.0.13', port=80,debug=True)

from pandas import DataFrame
a=[]
a1=[]
a2=[]
b=[0.4,0.55]
c=[0.66,0.157]
a1.extend(b)
a2.extend(c)
a.append(a1)
a.append(a2)
print(a)
a=DataFrame(a)
print(a)

# df2 = DataFrame([accXs, accYs, accZs, gyrXs, gyrYs, gyrZs], index=['accXs', 'accYs', 'accZs', 'gyrXs', 'gyrYs', 'gyrZs'])
# df2 = df2.stack()
# df2 = df2.unstack(0)
# n=len(accXs)
# Data = []
# for item in ['accXs', 'accYs', 'accZs', 'gyrXs', 'gyrYs', 'gyrZs']:
#     dat1 = []
#     for j in range(1, n):
#         dat = df2.loc[0, item]
#         dat = dat[j]
#         dat1.extend(dat)
#     Data.append(dat1)
#     print(Data)
