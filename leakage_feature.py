# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:21:32 2017

@author: yuwei
"""

import pandas as pd
import datetime as dt
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score


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
    dataset.Date_received = dataset.Date_received.map(lambda x : dt.datetime.strptime(str(x),"%Y%m%d") if x!='null' else x)
    #off_train:领券时期在20160515-20160615之间
    dataset2 = off_train[(off_train.Date_received >= '20160516') & (off_train.Date_received<='20160615')]
    #off_train:领券日期在20160501-20160530之间
    dataset1 = off_train[(off_train.Date_received>='20160501') & (off_train.Date_received<='20160531')]
    
    #对应的特征提取区间                     
    feature = off_train[((off_train.Date_received>='20160501')&(off_train.Date_received<='20160615'))]
    feature2 = off_train[((off_train.Date_received>='20160315')&(off_train.Date_received<='20160501'))]
    feature1 = off_train[((off_train.Date_received>='20160115')&(off_train.Date_received<='20160315'))]
                         
    #空闲的购买特征提取区间
    space = off_train[(off_train['Date']>='20160616')&(off_train['Date']!='null')]
    space1 = off_train[(off_train['Date']>='20160501')&(off_train['Date']<='20160515')]
    space2 = off_train[(off_train['Date']>='20160315')&(off_train['Date']<='20160330')]
                       
    return dataset,dataset1,dataset2,feature,feature1,feature2,space,space1,space2
    
    
def get_user_feat(dataset):
    "用户领券特征"
    
    "计算用户在当月领券次数--1"
    dataset['theMonth_receivedCount'] = dataset['User_id']
    feat = pd.pivot_table(dataset,index=['User_id'],values='theMonth_receivedCount',aggfunc='count').reset_index()
    del dataset['theMonth_receivedCount']
    result = pd.merge(dataset,feat,on=['User_id'],how='left')
    
    "计算用户在当月领券重复的总数--1"
    dataset['theMonth_receivedRepCount'] = dataset['Coupon_id']
    feat = pd.pivot_table(dataset,index=['User_id','Coupon_id'],values='theMonth_receivedRepCount',aggfunc='count').reset_index()
    del dataset['theMonth_receivedRepCount']
    result = pd.merge(result,feat,on=['User_id','Coupon_id'],how='left')
    #当月仅领取过一次的样本赋值为0
    result.theMonth_receivedRepCount = result.theMonth_receivedRepCount.map(lambda x: 0 if x==1 else x)
    
    "计算用户在领券当天领券总数--1"
    dataset['theDay_receivedCount'] = dataset['Date_received']
    feat = pd.pivot_table(dataset,index=['User_id','Date_received'],values='theDay_receivedCount',aggfunc='count').reset_index()
    del dataset['theDay_receivedCount']
    result = pd.merge(result,feat,on=['User_id','Date_received'],how='left')
    
    "计算用户在领券当天领券重复的总数--1"
    dataset['theDay_receivedRepCount'] = dataset['Coupon_id']
    feat = pd.pivot_table(dataset,index=['User_id','Coupon_id','Date_received'],values='theDay_receivedRepCount',aggfunc='count').reset_index()
    del dataset['theDay_receivedRepCount']
    result = pd.merge(result,feat,on=['User_id','Coupon_id','Date_received'],how='left')
    #当天仅领取过一次的样本赋值为0
    result.theDay_receivedRepCount = result.theDay_receivedRepCount.map(lambda x: 0 if x==1 else x)
    
    "计算用户领取特定商家的数目--1"
    dataset['userMerchant_count'] = dataset['Merchant_id']
    feat = pd.pivot_table(dataset,index=['User_id','Merchant_id'],values='userMerchant_count',aggfunc='count').reset_index()
    del dataset['userMerchant_count']
    result = pd.merge(result,feat,on=['User_id','Merchant_id'],how='left')
    
    "计算用户领取的所有优惠券种类数目--1"
    feat = dataset[['User_id','Coupon_id']]
    #去除重复
    feat = feat.drop_duplicates(['User_id','Coupon_id'],keep='first')
    feat = pd.pivot_table(feat,index=['User_id'],values='Coupon_id',aggfunc='count').reset_index()
    feat.columns = ['User_id','cate_count']
    result = pd.merge(result,feat,on=['User_id'],how='left')
    
    "计算每个商户共发行多少种优惠券--1"
    feat = dataset[['Merchant_id','Coupon_id']]
    feat = feat.drop_duplicates(['Merchant_id','Coupon_id'],keep='first')
    feat = pd.pivot_table(feat,index=['Merchant_id'],values='Coupon_id',aggfunc='count').reset_index()
    feat.columns = ['Merchant_id','MerchantCoupon_count']
    result = pd.merge(result,feat,on=['Merchant_id'],how='left')
    
    "计算是否是领取特定优惠券首次和末次--2"
    time = dataset[['User_id','Coupon_id','Date_received']]
    #转为字符串
    time.Date_received = time.Date_received.astype('str')
    time = time.groupby(['User_id','Coupon_id'])['Date_received'].agg(lambda x:':'.join(x)).reset_index()
    time['receive_number'] = time.Date_received.apply(lambda s:len(s.split(':')))
    t = time[time.receive_number>1]
    tb = time[time.receive_number==1];tb['receive_first'] = -1
    t['receive_number_max'] = t.Date_received.map(lambda s:max(s.split(':')))
    t['receive_number_min'] = t.Date_received.map(lambda s:min(s.split(':')))
    del t['Date_received'];del t['receive_number'];del tb['receive_number']
    #第一次领取
    t1 = t[['User_id','Coupon_id','receive_number_min']]
    t1 = t1.rename(columns={'receive_number_min':'Date_received'})
    t1['receive_first'] = 1
    #合并上只领取过一次的情况
    t1 = t1.append(tb)
    t1.Date_received = t1.Date_received.map(lambda x : dt.datetime.strptime(x,"%Y-%m-%d"))
    #最后一次领取
    t2 = t[['User_id','Coupon_id','receive_number_max']]
    t2 = t2.rename(columns={'receive_number_max':'Date_received'})
    t2['receive_last'] = 1
    #合并上只领取过一次的情况
    tb = tb.rename(columns={'receive_first':'receive_last'})
    t2 = t2.append(tb)
    t2.Date_received = t2.Date_received.map(lambda x : dt.datetime.strptime(x,"%Y-%m-%d"))
    result = pd.merge(result,t1,how='left')
    result = pd.merge(result,t2,how='left')
    result = result.fillna(0)
    
    "计算用户领取了多少个不同商户的优惠券--1"
    "计算商户被多少不同的用户领取过优惠券--1"
    #取用户和商户信息
    t = dataset[['User_id','Merchant_id']]
    #去除重复的商家用户对
    t = t.drop_duplicates(subset=['User_id','Merchant_id'], keep = 'first')
    #将商户转为字符串
    t.Merchant_id = t.Merchant_id.astype('str')
    "计算用户领取了多少个不同商户的优惠券"
    #按用户分组，连接所有商户
    t1 = t.groupby(['User_id'])['Merchant_id'].agg(lambda x:','.join(x)).reset_index()
    #统计用户领取商户的个数
    t1['merchant_receivedCount'] = t1.Merchant_id.apply(lambda s:len(s.split(',')))
    del t1['Merchant_id']
    result = pd.merge(result,t1,on=['User_id'],how='left')
    "计算商户被多少不同的用户领取过优惠券"
    t.Merchant_id = t.Merchant_id.astype('int64')
    t.User_id = t.User_id.astype('str')
    #按商户分组，连接所有用户
    t2 = t.groupby(['Merchant_id'])['User_id'].agg(lambda x:','.join(x)).reset_index()
    #统计用户领取商户的个数
    t2['user_receivedCount'] = t2.User_id.apply(lambda s:len(s.split(',')))
    del t2['User_id']
    result = pd.merge(result,t2,on=['Merchant_id'],how='left')
    
    "计算领券日期是一周/一月的第几天以及逆序--4"
    #领券日期是一周中的第几天
    result['order_in_weekday'] = 0
    result.order_in_weekday = result.Date_received.map(lambda x : x.weekday())
    result['order_in_weekday_inverse'] = 6 - result['order_in_weekday']
    #领券日期是一月中的第几天
    result['order_in_monthday'] = 0
    result.order_in_monthday = result.Date_received.map(lambda x : x.day)
    result['order_in_monthday_inverse'] = 31 - result['order_in_monthday']
    
    "计算商户被领取的优惠券数目--1"
    dataset['merchant_count'] = dataset['Merchant_id']
    feat = pd.pivot_table(dataset,index=['Merchant_id'],values='merchant_count',aggfunc='count').reset_index()
    del dataset['merchant_count']
    result = pd.merge(result,feat,on=['Merchant_id'],how='left')
    
    "计算当前领取的优惠券与上一次/下一次领券是否一样--2"
    #首先分离重复领券和未重复领券用户
    dataset['user_month_count'] = dataset['User_id']
    feat = pd.pivot_table(dataset,index=['User_id'],values='user_month_count',aggfunc='count').reset_index()
    del dataset['user_month_count']
    feat = pd.merge(dataset,feat,on=['User_id'],how='left')
    #对只领取过一次的用户进行处理
    feat_1 = feat[feat['user_month_count']==1]
    feat_1['next_is_same'] = -1;feat_1['last_is_same'] = -1
    feat_1 = feat_1[['User_id','Coupon_id','Date_received','next_is_same','last_is_same']]
    #对多次领券的用户进行处理
    feat_2 = feat[feat['user_month_count']>1]
    feat_2 = feat_2.sort_values(['User_id','Date_received'],ascending=True)
    feat_2.index = range(len(feat_2))
    #下一次领券
    feat_2['User_id_next'] = feat_2.shift(1).User_id;feat_2['Coupon_id_next'] = feat_2.shift(1).Coupon_id
    feat_2 = feat_2.assign(next_is_same = ((feat_2.User_id == feat_2.User_id_next) & (feat_2.Coupon_id == feat_2.Coupon_id_next)))
    feat_2.next_is_same = feat_2.next_is_same.map(lambda x: 1 if x==True else 0)
    #上一次领券
    feat_2['User_id_last'] = feat_2.shift(-1).User_id;feat_2['Coupon_id_last'] = feat_2.shift(-1).Coupon_id
    feat_2 = feat_2.assign(last_is_same = ((feat_2.User_id == feat_2.User_id_last) & (feat_2.Coupon_id == feat_2.Coupon_id_last)))
    feat_2.last_is_same = feat_2.last_is_same.map(lambda x: 1 if x==True else 0)
    feat_2 = feat_2[['User_id','Coupon_id','Date_received','next_is_same','last_is_same']]
    #合并
    feat = feat_2.append(feat_1)
    result = pd.merge(result,feat,how='left')
    
    return result
    
def get_userid_feat(feature,dataset):
    "提取feature区间的用户特征"
    
    #转化折扣率
    feature['discount_transfer'] = feature.Discount_rate.map(lambda x: (int(x.split(':')[1])/int(x.split(':')[0])) if ':' in x else x)
    feature['discount_transfer'] = feature['discount_transfer'].astype('float')
    #是否消费
    feature['buy_orNot'] = 0
    feature.Date = feature.Date.astype('str')
    feature.buy_orNot = feature.Date.map(lambda x : 0 if x=='null' else 1)
    #转化距离为float型
    feature.Distance = feature.Distance.map(lambda x :np.nan if x=='null' else int(x))
    
    "计算用户领券的次数--1"
    feature['receive_count_feat'] = feature['User_id']
    feat = pd.pivot_table(feature,index=['User_id'],values='receive_count_feat',aggfunc='count').reset_index()
    del feature['receive_count_feat']
    result_feat = pd.merge(dataset,feat,on=['User_id'],how='left')
    
    "计算用户领券后消费的次数--1"
    feature['buy_count_feat'] = feature['buy_orNot']
    feat = pd.pivot_table(feature,index=['User_id'],values='buy_count_feat',aggfunc='sum').reset_index()
    result_feat = pd.merge(result_feat,feat,on=['User_id'],how='left')
    del feature['buy_count_feat']

    "计算用户领券后未消费的次数--1"
    result_feat['no_buy_count_feat'] = result_feat['receive_count_feat'] - result_feat['buy_count_feat']

    "计算用户领券后进行消费率--1"
    result_feat['buy_receive_ratio_feat'] = result_feat['buy_count_feat']/result_feat['receive_count_feat']

    "计算用户消费过的折扣率的平均/最高/最低值--3"
    #平均值
    feat = feature[feature['buy_orNot']==1]
    feat['buy_mean_discount'] = feat['discount_transfer']
    feat = pd.pivot_table(feat,index=['User_id'],values='buy_mean_discount',aggfunc='mean').reset_index()
    result_feat = pd.merge(result_feat,feat,on=['User_id'],how='left')
    #最大值
    feat = feature[feature['buy_orNot']==1]
    feat['buy_max_discount'] = feat['discount_transfer']
    feat = pd.pivot_table(feat,index=['User_id'],values='buy_max_discount',aggfunc='max').reset_index()
    result_feat = pd.merge(result_feat,feat,on=['User_id'],how='left')
    #最小值
    feat = feature[feature['buy_orNot']==1]
    feat['buy_min_discount'] = feat['discount_transfer']
    feat = pd.pivot_table(feat,index=['User_id'],values='buy_min_discount',aggfunc='min').reset_index()
    result_feat = pd.merge(result_feat,feat,on=['User_id'],how='left')
    #对于消费次数为0的情况
    nobuy = list(map(lambda x,y : -1 if x == 0 else y,result_feat.buy_count_feat,list(result_feat.buy_mean_discount)));result_feat.buy_mean_discount = nobuy
    nobuy = list(map(lambda x,y : -1 if x == 0 else y,result_feat.buy_count_feat,list(result_feat.buy_max_discount)));result_feat.buy_max_discount = nobuy
    nobuy = list(map(lambda x,y : -1 if x == 0 else y,result_feat.buy_count_feat,list(result_feat.buy_min_discount)));result_feat.buy_min_discount = nobuy
    
    "计算消费过的不同商家的次数--1"
    feat = feature[feature['buy_orNot']==1]
    feat = feat.drop_duplicates(['User_id','Merchant_id'],keep='first')
    feat['buy_DiffMerchant_count'] = feat['Merchant_id']
    feat = pd.pivot_table(feat,index=['User_id'],values='buy_DiffMerchant_count',aggfunc='count').reset_index()
    result_feat = pd.merge(result_feat,feat,on=['User_id'],how='left')
    nobuy = list(map(lambda x,y : 0 if x == 0 else y,result_feat.buy_count_feat,list(result_feat.buy_DiffMerchant_count)));result_feat.buy_DiffMerchant_count = nobuy
    
    "计算消费过不同商家次数比上总的不同的商家数--1"
    sum_merchant = len(set(feature['Merchant_id']))
    result_feat['buy_DiffMerchant_count_merchant_ratio'] = result_feat['buy_DiffMerchant_count'] / sum_merchant

    "计算消费过的不同消费券的次数--1"
    feat = feature[feature['buy_orNot']==1]
    feat = feat.drop_duplicates(['User_id','Coupon_id'],keep='first')
    feat['buy_DiffCoupon_count'] = feat['Coupon_id']
    feat = pd.pivot_table(feat,index=['User_id'],values='buy_DiffCoupon_count',aggfunc='count').reset_index()
    result_feat = pd.merge(result_feat,feat,on=['User_id'],how='left')
    nobuy = list(map(lambda x,y : 0 if x == 0 else y,result_feat.buy_count_feat,list(result_feat.buy_DiffCoupon_count)));result_feat.buy_DiffCoupon_count = nobuy
    
    "计算消费过不同商家次数比上总的不同的商家数--1"
    sum_coupon = len(set(feature['Coupon_id']))
    result_feat['buy_DiffCoupon_count_coupon_ratio'] = result_feat['buy_DiffCoupon_count'] / sum_coupon

    "用户平均消费在每个商家的次数--1"
    result_feat['buy_merchant_count_mean'] = result_feat['buy_count_feat'] / result_feat['buy_DiffMerchant_count']
    nobuy = list(map(lambda x,y : 0 if x == 0 else y,result_feat.buy_count_feat,list(result_feat.buy_merchant_count_mean)));result_feat.buy_merchant_count_mean = nobuy

    "用户消费券的平均/最大/最小距离--3"
    #平均距离
    feat = feature[feature['buy_orNot']==1]
    feat['buy_mean_distance'] = feat['Distance']
    feat = pd.pivot_table(feat,index=['User_id'],values='buy_mean_distance',aggfunc='mean').reset_index()
    result_feat = pd.merge(result_feat,feat,on=['User_id'],how='left')
    nobuy = list(map(lambda x,y : -1 if x == 0 else y,result_feat.buy_count_feat,list(result_feat.buy_mean_distance)));result_feat.buy_mean_distance = nobuy
    #最大距离
    feat = feature[feature['buy_orNot']==1]
    feat['buy_max_distance'] = feat['Distance']
    feat = pd.pivot_table(feat,index=['User_id'],values='buy_max_distance',aggfunc='max').reset_index()
    result_feat = pd.merge(result_feat,feat,on=['User_id'],how='left')
    nobuy = list(map(lambda x,y : -1 if x == 0 else y,result_feat.buy_count_feat,list(result_feat.buy_max_distance)));result_feat.buy_max_distance = nobuy
    #最小距离
    feat = feature[feature['buy_orNot']==1]
    feat['buy_min_distance'] = feat['Distance']
    feat = pd.pivot_table(feat,index=['User_id'],values='buy_min_distance',aggfunc='min').reset_index()
    result_feat = pd.merge(result_feat,feat,on=['User_id'],how='left')
    nobuy = list(map(lambda x,y : -1 if x == 0 else y,result_feat.buy_count_feat,list(result_feat.buy_min_distance)));result_feat.buy_min_distance = nobuy
    
    #########提取商户特征###########
#    "计算商户被领券的次数--1"
#    feature['merchant_received_count_featM'] = feature['User_id']
#    feat = pd.pivot_table(feature,index=['Merchant_id'],values='merchant_received_count_featM',aggfunc='count').reset_index()
#    del feature['merchant_received_count_featM']
#    result_feat = pd.merge(result_feat,feat,on=['Merchant_id'],how='left')
#    
#    "计算商户被领券后消费的次数--1"
#    feature['buy_merchant_count_featM'] = feature['buy_orNot']
#    feat = pd.pivot_table(feature,index=['Merchant_id'],values='buy_merchant_count_featM',aggfunc='sum').reset_index()
#    del feature['buy_merchant_count_featM']
#    result_feat = pd.merge(result_feat,feat,on=['Merchant_id'],how='left')
#    
#    "计算商户领券后不被消费的次数--1"
#    result_feat['no_buy_merchant_count_featM'] = result_feat['merchant_received_count_featM'] - result_feat['buy_merchant_count_featM']
#    
#    "计算商户被领券后消费的概率"
#    result_feat['buy_merchant_receive_ratio'] = result_feat['buy_merchant_count_featM'] / result_feat['merchant_received_count_featM']
    
#    "计算商户被消费过的折扣率的平均/最高/最低值--3"
#    #平均值
#    feat = feature[feature['buy_orNot']==1]
#    feat['buy_mean_discount_merchant'] = feat['discount_transfer']
#    feat = pd.pivot_table(feat,index=['Merchant_id'],values='buy_mean_discount_merchant',aggfunc='mean').reset_index()
#    result_feat = pd.merge(result_feat,feat,on=['Merchant_id'],how='left')
#    #最大值
#    feat = feature[feature['buy_orNot']==1]
#    feat['buy_max_discount_merchant'] = feat['discount_transfer']
#    feat = pd.pivot_table(feat,index=['Merchant_id'],values='buy_max_discount_merchant',aggfunc='max').reset_index()
#    result_feat = pd.merge(result_feat,feat,on=['Merchant_id'],how='left')
#    #最小值
#    feat = feature[feature['buy_orNot']==1]
#    feat['buy_min_discount_merchant'] = feat['discount_transfer']
#    feat = pd.pivot_table(feat,index=['Merchant_id'],values='buy_min_discount_merchant',aggfunc='min').reset_index()
#    result_feat = pd.merge(result_feat,feat,on=['Merchant_id'],how='left')
#    #对于消费次数为0的情况
#    nobuy = list(map(lambda x,y : -1 if x == 0 else y,result_feat.buy_count_feat,list(result_feat.buy_mean_discount_merchant)));result_feat.buy_mean_discount_merchant = nobuy
#    nobuy = list(map(lambda x,y : -1 if x == 0 else y,result_feat.buy_count_feat,list(result_feat.buy_max_discount_merchant)));result_feat.buy_max_discount_merchant = nobuy
#    nobuy = list(map(lambda x,y : -1 if x == 0 else y,result_feat.buy_count_feat,list(result_feat.buy_min_discount_merchant)));result_feat.buy_min_discount_merchant = nobuy

    return result_feat

def model_xgb(train,test):
    train_y = train['label'].values
    train_x = train.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Distance',
                          'Date_received', 'Date','label'],axis=1).values
    test_x = test.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Distance',
                          'Date_received'],axis=1).values

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eta': 0.03,
              'max_depth': 6,  # 6
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 18  # 2
              }
    # 训练
    bst = xgb.train(params, dtrain, num_boost_round=1500)
    # 预测
    predict = bst.predict(dtest)
    test_xy = test[['User_id','Coupon_id','Date_received']]
    test_xy['prob'] = predict
    predict = pd.DataFrame(predict)
    predicted = pd.concat([test_xy[['User_id','Coupon_id','Date_received']], predict], axis=1)
    predicted = predicted.sort_values(0, ascending=True)
    #将日期转为字符串格式
    predicted.Date_received = predicted.Date_received.map(lambda x : dt.datetime.strftime(pd.to_datetime(x),"%Y%m%d"))
    predicted = predicted.drop_duplicates(subset=['User_id','Coupon_id','Date_received'], keep = 'first')
    return predicted
    
def train_model(dataset1,dataset2,dataset,feature1,feature2,feature):
   "调用模型"
   dataset1 = get_user_feat(dataset1);dataset1 = get_userid_feat(feature1,dataset1)
   dataset2 = get_user_feat(dataset2);dataset2 = get_userid_feat(feature2,dataset2)
   train = dataset1.append(dataset2)
   test = get_user_feat(dataset);test = get_userid_feat(feature,test)
   answer = model_xgb(train,test)
   return answer
   
def validateModel_xgb(train,test):
    train_y = train['label'].values
    train_x = train.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Distance',
                          'Date_received', 'Date','label','merchant_receivedCount'],axis=1).values
    test_x = test.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Distance',
                        'Date_received','Date','label','merchant_receivedCount'],axis=1).values
    #'merchant_receivedCount'
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eta': 0.03,
              'max_depth': 5,  # 6
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 18  # 2
              }
    # 训练
    bst = xgb.train(params, dtrain, num_boost_round=1500)
    # 预测
    predict = bst.predict(dtest)
    test_xy = test[['User_id','Coupon_id','Date_received','label']]
    test_xy['prob'] = predict
    predict = pd.DataFrame(predict)
    predicted = pd.concat([test_xy[['User_id','Coupon_id','Date_received','label']], predict], axis=1)
    predicted = predicted.sort_values(0, ascending=False)
    #将日期转为字符串格式
    predicted.Date_received = predicted.Date_received.map(lambda x : dt.datetime.strftime(pd.to_datetime(x),"%Y%m%d"))
    predicted = predicted.drop_duplicates(subset=['User_id','Coupon_id','Date_received'], keep = 'first')
    return predicted
    
def validate_model(feature1,feature2,dataset1,dataset2):
   "调用模型"
   dataset1 = get_user_feat(dataset1);dataset1 = get_userid_feat(feature1,dataset1)
   dataset2 = get_user_feat(dataset2);dataset2 = get_userid_feat(feature2,dataset2)
   
   train = dataset1
   test = dataset2
   answer = validateModel_xgb(train,test)
   #模型评估
   answer.columns = ['User_id', 'Coupon_id', 'Date_received', 'label', 'pre']
   answer['pre_label'] = 0
   answer.pre_label = answer.pre.map(lambda x: 1 if x >=0.5 else 0)
   print('\n'+'auc值为='+str(accuracy_score(answer.label, answer.pre_label)))
   return answer
   
    
if __name__ == '__main__':
    # data原始数据
   off_train,off_test = loadData()
   #划分数据集
   dataset,dataset1,dataset2,feature,feature1,feature2,space,space1,space2 = spiltData(off_train,off_test)
   #给训练集和验证集打标
   dataset1 = makeLabel(dataset1); dataset2 = makeLabel(dataset2)
   
#   #label数据集特征提取测试
#   feat = get_user_feat(dataset1)
   
#   feature数据集特征提取测试
#   dataset = get_user_feat(dataset)
#   userid_feat = get_userid_feat(feature,dataset)

   #跑测试集模型
   test_answer = train_model(dataset1,dataset2,dataset,feature1,feature2,feature)
   test_answer.to_csv('7.csv',index=False)
    
#   #跑验证集模型
#   validate_answer = validate_model(feature1,feature2,dataset1,dataset2)


    
def test_model(dataset,dataset1,dataset2,feature,feature1,feature2):
	 label_feat1 = get_label_feat(dataset1);
   featureUser1 = get_feature_userid_feat(feature1,dataset1)
   featureMerchant1 = get_feature_merchantid_feat(feature1,dataset1)
   train_1 = pd.merge(label_feat1,featureUser1,on='User_id',how='left');train_1 = pd.merge(train_1,featureMerchant1,on='Merchant_id',how='left')
   
   label_feat2 = get_label_feat(dataset2);
   featureUser2 = get_feature_userid_feat(feature2,dataset2)
   featureMerchant2 = get_feature_merchantid_feat(feature2,dataset2)
   train_2 = pd.merge(label_feat2,featureUser2,on='User_id',how='left');train_2 = pd.merge(train_2,featureMerchant2,on='Merchant_id',how='left')
   
   train = train_1.append(train_2)
	
	 label_feat = get_label_feat(dataset);
   featureUser = get_feature_userid_feat(feature,dataset)
   featureMerchant = get_feature_merchantid_feat(feature,dataset)
   test = pd.merge(label_feat,featureUser,on='User_id',how='left');test = pd.merge(test,featureMerchant,on='Merchant_id',how='left')
   answer = model_xgb(train,test)
   retrun answer
