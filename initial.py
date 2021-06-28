import pandas as pd
import numpy as np

# initial setting

class Account(dict): 
    '''
    账户对象
    该对象为一个dict
    keys为币对名称 
    value为一个Trade对象，内部记录了该币对的详细交易信息
    '''
    def __init__(self, trade_data, trade_ret, pairs, cost = 0.0004, weights = 'default'):
        if weights == 'default':
            self.weights = np.full((len(pairs), 1),1/len(pairs)) # 生成一个n*1的np.Narray 每个里面都是币对的初始weights
        else:
            self.weights = weights
        self.trade_data = trade_data
        self.trade_ret = trade_ret
        self.pairs = pairs
        self.cost = cost

        for pair in pairs:
            self[pair] = Trade(trade_ret, pair, cost = cost) # 初始化每一对pair 没有交易是怎么搞到下面的
    
    def get_ret_table(self):
        result = pd.DataFrame()
        for i in self.keys(): #将每一个币对的收益装在result中
            result[i] = self[i].ret_series
        result.index = self.trade_data.index
        return result
    
    def get_ret_series(self):
        result = self.get_ret_table()
        ret_series = np.dot(result, self.weights).reshape(len(result),)
        return pd.Series(ret_series, index = result.index)

class Trade(): #对象是每一个单独的币对
    
    # 对交易对象初始化
    def __init__(self, ret_data, pair, cost = 0.0004): # 输入的ret_data一定要是time为index的
        self.per = 1 #默认仓位为1 
        self.lcoin = None  # 当前多头 
        self.scoin = None # 当前空头
        self.state = 0 # 当前账户状态 是否为0
        self.count = 0 # 交易次数
        self.cost = cost # 交易费用
        self.pair = pair # 当前交易币对
        self.ret_series = [] # 全局收益序列
        self.holding_ret = [] # 当前币对的收益率序列
        self.ret_data = ret_data
        self.summary = {'count':[], 'begin_date':[], 'end_date':[], 'long':[], 'short':[], 'ret':[]}

    def open(self, i, direction, per=1, rev=False):  #开仓的时候 signal指的是第i个时间的signal
        # rev是指是不是反向开的仓是的话再减掉一个关仓的交易费用
        if direction == 1: #开仓信号为1，那么多1空0
            self.lcoin = self.pair[1]
            self.scoin = self.pair[0]
        if direction == -1: # 开仓信号为-1，那么多0空1
            self.lcoin = self.pair[0]
            self.scoin = self.pair[1]
        ret = (self.ret_data[self.lcoin][i] - self.ret_data[self.scoin][i])/2-self.cost  # /2的原因是当仓位是1的时候一半做多一半做空，手续费其实是单边千四，cost/2*2导致开一次是千四的手续费
        if rev == True:
            ret = ret - self.cost
        self.state = direction
        self.count = self.count+1
        self.summary['count'].append(self.count)
        self.summary['begin_date'].append(i)
        self.summary['long'].append(self.lcoin)
        self.summary['short'].append(self.scoin)
        self.ret_series.append(ret*per)
        self.holding_ret.append(ret*per)
        # self.summary['ret'].append(self.holding_ret) 为什么只在关仓的时候ret加在ret中
        self.per = per 
    
    def close(self,i):
        self.state = 0
        self.holding_ret.append(-self.cost*self.per)
        self.ret_series.append(-self.cost*self.per)
        self.summary['end_date'].append(i)
        self.summary['ret'].append(self.holding_ret) 
        self.holding_ret = [] #让币对的收益率序列重置为0
        self.per = 1

    def hold(self,i):
        ret = (self.ret_data[self.lcoin][i] - self.ret_data[self.scoin][i])/2
        self.holding_ret.append(ret*self.per)
        self.ret_series.append(ret*self.per)
    
    def sleep(self,i): #即没达到开仓条件的时候
        self.state = 0
        self.ret_series.append(0)
    
    def get_summary(self): # return得到的是summary的dataframe
        a = pd.DataFrame(self.summary)
        a['sum_ret'] = a['ret'].apply(np.sum) 
        a['hold_time'] = (a['end_date']-a['begin_date'])*5 #这里的5指的是5min一个bar
        a['win&loss'] = (a['sum_ret']>0).astype(int) #找出sumret中大于0的有多少
        return a 
    
    def win_rate(self):
        return self.get_summary()['win&loss'].mean() # 为什么是用sumret的数据来得到胜率？？不应该用每一期的ret大于0小于0来判断么 而且为什么胜率用的是mean 待后续考察
