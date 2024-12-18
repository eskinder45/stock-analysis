import pandas as pd
import talib
import numpy as np



class technicalIndicators:
    #Simple Moving Average
    def simple_moving_avarage(self,df:pd.DataFrame):
        # 20-day and 50-day SMA
        df['SMA20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA50'] = talib.SMA(df['Close'], timeperiod=50)
        return df
    # exponential moving averate
    def exponential_moving_average(self,df:pd.DataFrame):
        # 20-day EMA
        df['EMA20'] = talib.EMA(df['Close'], timeperiod=20)
        return df
    #Bollinger Bands
    def bollinger_band(self,df:pd.DataFrame):
        df['UpperBB'], df['MiddleBB'], df['LowerBB'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        return df 
    #Relative Strength Index
    def relative_strength_index(self,df:pd.DataFrame):
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        return df
    #MA convergence divergence
    def ma_convergence_divergence(self,df:pd.DataFrame):
        df['MACD'], df['MACDSignal'], df['MACDHist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        return df
    #Avg true range
    def avg_true_range(self,df:pd.DataFrame):
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        return df

