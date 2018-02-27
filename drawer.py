# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:35:09 2017

@author: yuwei
"""
import pandas as pd
import datetime as dt
#python画图包      
import matplotlib.pyplot as plt


test_revised = 'ccf_offline_stage1_test_revised.csv'
offline_train = 'ccf_offline_stage1_train.csv'
online_train = 'ccf_online_stage1_train.csv'


def loadData():
    "传入数据时间化"
    dataOffline = pd.read_csv(offline_train)
    dataOffline = dataOffline[dataOffline['Date_received']!='null']
    dataOffline.Date = dataOffline.Date.map(lambda x : dt.datetime.strptime(x,"%Y%m%d") if x!='null' else x)
    dataOffline.Date_received = dataOffline.Date_received.map(lambda x : dt.datetime.strptime(x,"%Y%m%d") if x!='null' else x)
    return dataOffline


def drawer(data):
    #绘制每天领券次数与天数关系
    coupon_count = data.groupby(data['Date_received']).size()
    coupon_count_X = coupon_count.index
    coupon_count_Y = coupon_count
    fig1 = plt.figure()
    plt.bar(coupon_count_X,coupon_count_Y,fc='g')
    fig1.show()
    
    #绘制每天消费次数与天数关系
    buy_count = data[data['Date']!='null']
    buy_count = buy_count.groupby(data['Date']).size()
    buy_count_X = buy_count.index
    buy_count_Y = buy_count
    fig2 = plt.figure()
    plt.bar(buy_count_X,buy_count_Y,fc='b')
    fig2.show()
    
    #绘制折扣率与消费次数关系
    discount_count = data.groupby(data['Discount_rate']).size()
    discount_count_X = discount_count.index
    discount_count_Y = discount_count
    fig3 = plt.figure()
    plt.bar(range(len(discount_count_X)),discount_count_Y,fc='r')
    plt.xticks(range(discount_count_X.size),discount_count_X, rotation=40)
    fig3.show()
    
if __name__ == '__main__':
#    dataOffline,dataOnline,dataTest = loadData()
    dataOffline = loadData()
    drawer(dataOffline)
    















