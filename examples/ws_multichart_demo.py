'''
This file contains a simple multi chart animation demo using mplfinance.

As described in 'renkodf_modes.ipynb'
we can have multiple dataframes of different modes from the same instance.

It is highly recommended that, in real cases, the Animation/Real time Renko be run
in another process using the multiprocessing library of your choice,
as the processing load required to generate the dataframe may cause bottlenecks
in other services such as websocket connection, API requests , etc.

Taking Python's multiprocessing library as an example, the basic usage would be:
* mp.Process to run all this code (and its indicators, signals, etc)
* mp.Array to update and read timestamp/price value.
'''

import mplfinance as mpf
import pandas as pd
from matplotlib import animation

from renkodf import RenkoWS

df_ticks = pd.read_parquet('data/BNBUSDT-aggTrades-2023-06_9000Rows.parquet')

initial_timestamp = df_ticks['timestamp'].iat[0]
initial_price = df_ticks['close'].iat[0]

r = RenkoWS(initial_timestamp, initial_price, brick_size=0.04)
initial_df = r.initial_df

fig = mpf.figure(style='charles', figsize=(12,9))
fig.subplots_adjust(hspace=0.01, wspace=0.01)

axes = [fig.add_subplot(2,3,1),
        fig.add_subplot(2,3,2),
        fig.add_subplot(2,3,3)]

avs = [fig.add_subplot(3,3,7,sharex=axes[0]),
       fig.add_subplot(3,3,8,sharex=axes[1]),
       fig.add_subplot(3,3,9,sharex=axes[2])]

mpf.plot(initial_df,type='candle',ax=axes[0],volume=avs[0],axtitle='Normal')
mpf.plot(initial_df,type='candle',ax=axes[1],volume=avs[1],axtitle='Wicks')
mpf.plot(initial_df,type='candle',ax=axes[2],volume=avs[2],axtitle='Nongap')

def animate(ival):
    if (0 + ival) >= len(df_ticks):
        print('no more data to plot')
        ani.event_source.interval *= 3
        if ani.event_source.interval > 12000:
            exit()
        return

    timestamp = df_ticks['timestamp'].iat[(0 + ival)]
    price = df_ticks['close'].iat[(0 + ival)]

    r.add_prices(timestamp, price)

    df_normal = r.renko_animate('normal', max_len=100, keep=50)
    df_wicks = r.renko_animate('wicks', max_len=100, keep=50)
    df_nongap = r.renko_animate('nongap', max_len=100, keep=50)

    for ax in axes:
        ax.clear()
    for av in avs:
        av.clear()

    mpf.plot(df_normal, type='candle', ax=axes[0], volume=avs[0], axtitle='Normal')
    mpf.plot(df_wicks, type='candle', ax=axes[1], volume=avs[1], axtitle='Wicks')
    mpf.plot(df_nongap, type='candle', ax=axes[2], volume=avs[2], axtitle='Nongap')


ani = animation.FuncAnimation(fig, animate, interval=80)
mpf.show()
