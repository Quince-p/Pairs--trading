import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from sklearn.cluster import affinity_propagation
from sklearn.covariance import GraphicalLassoCV
'''
contain：
get_pairs- 获取所有币对
choose_pairsap：先聚类之后按照三种方法来选择币对
choose_pairs：不聚类直接按照三种方法选择币对
输入：
ob_data:标准化价格数据
keep:保留的币对个数
by：选择币对的方法{'corr','coint','cumstd'}
'''

def get_pairs(coins):
    ## 获取所有的币对 -修改：防止columns中出现time的情况
    if 'time' in coins:
        coins.remove('time')
    a = []
    for i in range(len(coins)):
        for j in range(i+1,len(coins)):
            a.append((coins[i],coins[j]))
    return a

def coint_test(y, x):
    # 协整检验
    t_test, p_value, _ = coint(y, x)
    return t_test

def get_minstd(y,x):
    # 最小累计方差
    return -sum((y-x)**2)


def Clusters(df):
    if 'time' in df.columns:
        df.drop(['time'],axis=1,inplace=True)
    a=df.columns
    stock_dataset = np.array(df)
    stock_dataset = stock_dataset/np.std(stock_dataset,axis=0)
    np.nan_to_num(stock_dataset)
    stock_model = GraphicalLassoCV()
    stock_model.fit(stock_dataset)
    _,labels = affinity_propagation(stock_model.covariance_)
    n_labels = max(labels)
    labels=pd.DataFrame(labels)
    labels.index=a
    return labels

def choose_pairsap(ob_data,keep=10,by='cumstd'):
    ## keep:保留币对数量
    # by：coint,corr,mstd # 币对选取依据 
    corr = []
    t_test = []
    cumstd = []
    a=[]
    all_pairs = get_pairs(list(ob_data.columns))
    map=Clusters(ob_data)
    if by=='corr':
        for i in  all_pairs:
            coin0 = i[0]
            coin1 = i[1]
            if (map[map.index==coin0].values)==(map[map.index==coin1].values):
                b=1
            else:
                b=0
            a.append(b)
            corr.append(ob_data[[coin0,coin1]].corr().iloc[0,1])
        pairs_info = pd.DataFrame({'corr':corr},index=all_pairs)
        a=pd.DataFrame({'AP':a},index=all_pairs)
        pairs_info=pd.concat([pairs_info,a],axis=1)
    elif by=='coint':
        for i in  all_pairs:
            coin0 = i[0]
            coin1 = i[1]
            if (map[map.index==coin0].values)==(map[map.index==coin1].values):
                b=1
            else:
                b=0
            a.append(b)
            t=abs(coint_test(ob_data[coin0],ob_data[coin1]))
            t_test.append(t)
        pairs_info = pd.DataFrame({'coint':t_test},index=all_pairs)
        a=pd.DataFrame({'AP':a},index=all_pairs)
        pairs_info=pd.concat([pairs_info,a],axis=1)
    elif by=='cumstd':
        for i in  all_pairs:
            coin0 = i[0]
            coin1 = i[1]
            if (map[map.index==coin0].values)==(map[map.index==coin1].values):
                b=1
            else:
                b=0
            a.append(b)
            cumstd.append(get_minstd(ob_data[coin0],ob_data[coin1]))
        pairs_info = pd.DataFrame({'cumstd':cumstd},index=all_pairs)
        a=pd.DataFrame({'AP':a},index=all_pairs)
        pairs_info=pd.concat([pairs_info,a],axis=1)
    pairs_info=pairs_info[pairs_info['AP']==1]
    pairs_info = pairs_info.sort_values(by=[by],ascending=False)
    
   # pairs_info = pairs_info.sort_values(by=by,ascending=False)
    keep_pairs = list(pairs_info.head(keep).index)
    return keep_pairs

# 修改完成
def choose_pairs(ob_data, keep = 5, by = 'coint'):
    # keep：为保留币对数量
    # by 币对选择依据
    corr = []
    t_test = []
    cumstd = []
    # 可能需要修改 在每一个周期内需要删除一些币对
    all_pairs = get_pairs(list(ob_data.columns))
    if by == 'corr':
        for i in all_pairs:
            coin0 = i[0]
            coin1 = i[1]
            corr.append(ob_data[[coin0,coin1]].corr().iloc[0,1])
        pairs_info = pd.DataFrame({'corr': corr}, index = all_pairs)
    elif by == 'coint':
        for i in all_pairs:
            coin0 = i[0]
            coin1 = i[1]
            t_test.append(np.abs(coint(ob_data[coin0], ob_data[coin1])[0])) # 待若玮查看修改
        pairs_info = pd.DataFrame({'coint':t_test}, index = all_pairs)
    elif by == 'cumstd':
        for i in all_pairs:
            coin0 = i[0]
            coin1 = i[1]
            cumstd.append(get_minstd(ob_data[coin0], ob_data[coin1]))
        pairs_info = pd.DataFrame({'cumstd':cumstd}, index = all_pairs)
    else:
        print('Please enter correct method')
    pairs_info = pairs_info.sort_values(by = by, ascending=False)
    keep_pairs = list(pairs_info.head(keep).index)
    return keep_pairs