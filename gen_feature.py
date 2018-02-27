# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:21:32 2017

@author: yuwei
"""

import pandas as pd
import datetime as dt
import numpy as np


test_revised = 'ccf_offline_stage1_test_revised.csv'
offline_train = 'ccf_offline_stage1_train.csv'

def loadData():
    "读取数据集"
    off_test = pd.read_csv(test_revised)
    off_train = pd.read_csv(offline_train)
    return off_train,off_test

def makeLabel(data):
    "打标"
    #过滤掉其中接收优惠券日期为空的数据
    data = data[data['Date_received']!='null']
    #加入标签列
    data['label'] = 0
    #转换时间字符串格式为datetime格式
    data.Date = data.Date.map(lambda x : dt.datetime.strptime(x,"%Y%m%d") if x!='null' else x)
    data.Date_received = data.Date_received.map(lambda x : dt.datetime.strptime(x,"%Y%m%d") if x!='null' else x)
    data.index = range(len(data))
    datanull = data[data.Date=='null']
    datanotnull = data[data.Date!='null']
    datanotnull.index = range(len(datanotnull))
    datanotnull['day'] = 0
    datanotnull['day'] = pd.to_datetime(datanotnull['Date']) - pd.to_datetime(datanotnull['Date_received'])
    datanotnull.day = datanotnull.day.map(lambda x: x.days)
    datanotnull.label = datanotnull.day.map(lambda x: 1 if x <= 15 else 0)
    del datanotnull['day']
    data = pd.concat([datanull,datanotnull])
    data.index = range(len(data))
    return data
    
def spiltData(off_train,off_test):
    "划分数据集"
    #测试集
    dataset = off_test
    #off_train：领券时期在20160515-20160615之间的
    dataset2 = off_train[(off_train.Date_received >= '20160516') & (off_train.Date_received<='20160615')]
    #off_train： 领券日期在20160501-20160530的
    dataset1 = off_train[(off_train.Date_received>='20160501') & (off_train.Date_received<='20160530')]
                         
    #feature存放：筛出off_train中，以下四种情况：
    #消费日期data在20160516-20160630的，
    #或消费日期为空且领券日期在20160516-20160615的
    feature = off_train[
                         ((off_train.Date>='20160516')&(off_train.Date<='20160630'))
                         |((off_train.Date=='null')&(off_train.Date_received>='20160516')&(off_train.Date_received<='20160615'))]
    
    feature = feature[feature['Date_received']!='null']
    #转换时间字符串格式为datetime格式
    feature.Date = feature.Date.map(lambda x : dt.datetime.strptime(x,"%Y%m%d") if x!='null' else x)
    feature.Date_received = feature.Date_received.map(lambda x : dt.datetime.strptime(x,"%Y%m%d") if x!='null' else x)     
           
    #feature2存放：从off_train筛出：
    #消费日期在20160401-20160515的，
    #或领券日期在20160401-20160430但没有消费的
    feature2 = off_train[
                         (off_train.Date>='20160401') & (off_train.Date<='20160515')
                         |((off_train.Date=='null') & (off_train.Date_received>='20160401') & (off_train.Date_received<='20160430'))]
    feature2 = feature2[feature2['Date_received']!='null']
    #转换时间字符串格式为datetime格式
    feature2.Date = feature2.Date.map(lambda x : dt.datetime.strptime(x,"%Y%m%d") if x!='null' else x)
    feature2.Date_received = feature2.Date_received.map(lambda x : dt.datetime.strptime(x,"%Y%m%d") if x!='null' else x)     
    
    #feature1存放：从off_train筛出：
    #消费日期在20160316-20160430的，或
    #领券日期在20160316-20160415但没有消费的
    feature1 = off_train[
                         (off_train.Date>='20160316') & (off_train.Date<='20160430')
                         |((off_train.Date=='null') & (off_train.Date_received>='20160316') & (off_train.Date_received<='20160415'))]
    #转换时间字符串格式为datetime格式
    feature1 = feature1[feature1['Date_received']!='null']
    feature1.Date = feature1.Date.map(lambda x : dt.datetime.strptime(x,"%Y%m%d") if x!='null' else x)
    feature1.Date_received = feature1.Date_received.map(lambda x : dt.datetime.strptime(x,"%Y%m%d") if x!='null' else x)     
    
    return dataset,dataset1,dataset2,feature,feature1,feature2
    
    
def get_user_feat(data):
    "用户领券特征"
    #求用户前1,3,5,7,10,15,30领券总数
    uf = data[(np.max(data.Date_received) - data.Date_received) == dt.timedelta(days = 0)]
    uf['last' + str(1) + 'days_user_count'] = uf['User_id']
    result = pd.pivot_table(uf,index = 'User_id',values = ('last' + str(1) + 'days_user_count'),aggfunc = len).reset_index()
    for i in (0,2,4,6,9,14,29):
        uf = data[(np.max(data.Date_received) - data.Date_received) <= dt.timedelta(days = i + 1)]
        uf['last' + str(i + 2) + 'days_user_count'] = uf['User_id']
        pivot = pd.pivot_table(uf,index = 'User_id',values = ('last' + str(i + 2) + 'days_user_count'),aggfunc = len).reset_index()
        result = pd.merge(result,pivot,how = 'outer')
        print(i+1)
    final = pd.merge(data,result,on=['User_id'],how='left')
    #用户一周内领券次数的均值
    final['last8days_user_mean'] = final['last8days_user_count']/7
    

    #求用户前1,3,5,7,10,15,30领券set总数
    uf = data[(np.max(data.Date_received) - data.Date_received) == dt.timedelta(days = 0)]
    uf= uf.drop_duplicates(subset=['User_id','Coupon_id'], keep = 'first')
    uf['last' + str(1) + 'days_user_count_set'] = uf['User_id']
    result = pd.pivot_table(uf,index = 'User_id',values = ('last' + str(1) + 'days_user_count_set'),aggfunc = len).reset_index()
    for i in (0,2,4,6,9,14,29):
        uf = data[(np.max(data.Date_received) - data.Date_received) <= dt.timedelta(days = i + 1)]
        uf = uf.drop_duplicates(subset=['User_id','Coupon_id'], keep = 'first')
        uf['last' + str(i + 2) + 'days_user_count_set'] = uf['User_id']
        pivot = pd.pivot_table(uf,index = 'User_id',values = ('last' + str(i + 2) + 'days_user_count_set'),aggfunc = len).reset_index()
        result = pd.merge(result,pivot,how = 'outer')
        print(i+1)
    final = pd.merge(final,result,on=['User_id'],how='left')
    

    return final

    
if __name__ == '__main__':
    # data原始数据
   off_train,off_test = loadData()
   #划分数据集
   dataset,dataset1,dataset2,feature,feature1,feature2 = spiltData(off_train,off_test)
   #给训练集和验证集打标
   dataset1= makeLabel(dataset1); dataset2 = makeLabel(dataset2)
   uf = get_user_feat(feature)








