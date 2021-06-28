import pandas as pd
import numpy as np
import sys
sys.path.append('./Pairstrading/functions')
from initial import Account, Trade
from portfolio import getIVP,getClusterVar,getQuasiDiag,getRecBipart,correlDist
from choose_pairs import get_pairs, choose_pairs, choose_pairsap


def get_signal_updown(ob_data,test_data,pair,window,opn_up,opn_down,close_up,close_down): # 在修改的时候记住concat之后一定要drop掉重复，否则在pre_trade就会报错 
    '''    
    获取信号序列
    pair: 每一对trade_pairs
    window：滚动窗口
    opn,close: 信号阈值
    '''
    all_data = pd.concat((ob_data[list(pair)], test_data[list(pair)]),axis = 0) # 得到全样本的数据
    all_data = all_data[~all_data.index.duplicated()]
    all_data['spread'] = all_data[pair[0]] - all_data[pair[1]]
    # spread需要shift
    all_data['spread'] = all_data['spread'].shift(1)
    all_data['zscore'] = (all_data.spread - all_data.spread.rolling(window).mean())/all_data.spread.rolling(window).std()
    all_data['signal'] = np.where(all_data.zscore>=opn_up,2,0)
    all_data['signal'] = np.where((all_data.zscore>close_up)&(all_data.zscore<opn_up),1,all_data.signal)
    all_data['signal'] = np.where((all_data.zscore<close_down)&(all_data.zscore>opn_down),-1,all_data.signal)
    all_data['signal'] = np.where((all_data.zscore<=opn_down),-2,all_data.signal)
    all_data['direction'] = np.where(all_data.zscore>0,1,0)
    all_data['direction'] = np.where(all_data.zscore<0,-1,all_data.direction)
    signal = all_data['signal'][test_data.index]
    direction = all_data['direction'][test_data.index]
    return all_data, direction, signal


def single_trade_updown(acc,pair,ob_data,test_data,window=10,stop_loss=-1,opn_up=1.5,opn_down=-1.5,close_up=0.5,close_down=-0.5):
    trade1 = acc[pair] ## 初始化交易对象
    # all_data, direction, signal = get_signal(ob_data,test_data,pair,window=window,opn=opn ,close=close) # 获取该币对在train时期的信号注意这里可以用两个ob_data
    # opn,close = get_opnandclose(ob_data,pair,window)
    all_data, direction, signal = get_signal_updown(ob_data,test_data,pair,window=window,opn_up=opn_up,opn_down=opn_down,close_up=close_up,close_down=close_down)
    for i in signal.index:
        # 1. 第一期不开仓
        if i == signal.index[0]:
            trade1.sleep(i)
        # 2. 期末，如果有仓位则强制平仓 最后一期如果有仓位强制关仓
        elif i == signal.index[-1]:
            if  trade1.state==0:
                trade1.sleep(i)
            else:
                trade1.close(i)
        # 3. 既不是第一期也不是最后一期的情况
        else:
            if trade1.state == 0: # 若空仓
                if signal[i] == 2:
                    trade1.open(i,direction[i])
                elif signal[i] == -2:
                    trade1.open(i,direction[i])
                else:
                    trade1.sleep(i)
            elif trade1.state == 1:
                if np.sum(trade1.holding_ret)<stop_loss:
                    trade1.close(i)# 若收益低于止损线，则平仓
                else:
                    if signal[i] == -1:
                        trade1.close(i)
                    elif signal[i] == -2:
                        trade1.open(i,direction[i],rev=True)
                    else:
                        trade1.hold(i)
            elif trade1.state == -1:
                if np.sum(trade1.holding_ret)<stop_loss:
                    trade1.close(i)# 若收益低于止损线，则平仓
                else:
                    if signal[i] == 1:
                        trade1.close(i)
                    elif signal[i] == 2:
                        trade1.open(i,direction[i],rev=True)
                    else:
                        trade1.hold(i)
    return acc

def pre_trade(ob_data,ob_ret,trade_pairs,window=10,stop_loss=-1, cost=0.0004,opn_up=1.5,opn_down=-1.5,close_up=0.5,close_down=-0.5):
    acc = Account(ob_data, ob_ret, trade_pairs, cost=cost) #对train时期建立资金，对每个币对建立trade对象，默认对所有币对平分资金
    for pair in trade_pairs:  # 循环对每一个币开始交易
        # fig,ax = plt.subplots()
        # ax.plot(ob_data.index, ob_data[pair[0]], label = pair[0])
        # ax.plot(ob_data.index, ob_data[pair[1]], label = pair[1])
        # plt.suptitle('normal Price in train')
        # plt.show()
        acc = single_trade_updown(acc,pair,ob_data,ob_data,window=window,stop_loss =stop_loss,opn_up=opn_up,opn_down=opn_down,close_up=close_up,close_down=close_down)
        print(pair,'  sum_ret:  ',np.sum(acc[pair].ret_series))
    return acc.get_ret_series(), acc.get_ret_table(),acc

def get_weights(data, ret, trade_pairs, opn=1.5, close=0.5, window=10,cost=0.0004,stop_loss=-1,opn_up=1.5,opn_down=-1.5,close_up=0.5,close_down=-0.5): # 
    ret_series, ret_table, acc = pre_trade(data, ret, trade_pairs, window =window, cost = cost, stop_loss = stop_loss,opn_up=opn_up,opn_down=opn_down,close_up=close_up,close_down=close_down)
    cov, corr = ret_table.cov(), ret_table.corr() 
    dist=correlDist(corr)
    link=sch.linkage(dist,'single')
    sortIx=getQuasiDiag(link)
    sortIx=corr.index[sortIx].tolist() # recover labels
    df0=corr.loc[sortIx,sortIx] # reorder
    hrp=getRecBipart(cov,sortIx)
    weights = hrp[:, np.newaxis]
    return weights

def strategy(ob_data,ob_ret,test_data,test_ret,keep=5,window=10,stop_loss=-1,cost=0.0004,opn_up=1.5,opn_down=-1.5,close_up=0.5,close_down=-0.5):
    # 基本的交易策略
    # ob_data,ob_ret,test_data,test_ret 基本的数据输入 data代表标准化后的价格
    # return: ret_series,trade_summary,acc交易账户
    trade_pairs = choose_pairs(ob_data, keep = keep, by = 'cumstd')
    # trade_pairs = choose_pairsap(ob_data, keep = keep, by = 'cumstd')
    # print(trade_pairs)
    weights = get_weights(ob_data,ob_ret,trade_pairs,window=window,cost=cost,stop_loss=stop_loss,opn_up=opn_up,opn_down=opn_down,close_up=close_up,close_down=close_down)
    acc = Account(test_data,test_ret, trade_pairs,cost=cost,weights=weights)# 建立账户，对每个币对建立Trade对象，默认对所有币对平分资金
    for pair in trade_pairs:  # 循环对每一个币开始交易
        # fig,ax = plt.subplots()
        # ax.plot(test_data.index, test_data[pair[0]], label = pair[0])
        # ax.plot(test_data.index, test_data[pair[1]], label = pair[1])
        # plt.suptitle('normal Price in test')
        # plt.show()
        acc = single_trade_updown(acc,pair,ob_data,test_data,window=window,stop_loss =stop_loss,opn_up=opn_up,opn_down=opn_down,close_up=close_up,close_down=close_down)
        print(pair,'  sum_ret:  ',np.sum(acc[pair].ret_series))
    return acc.get_ret_series(),acc.get_ret_table(),acc

    