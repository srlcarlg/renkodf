'''
This file contains a simple animation demo using mplfinance.

It's highly recommended that, in real cases, the Animation/Real-Time Renko Chart
be running in another process using the multiprocessing library of your choice,
as the processing load required to generate the dataframe may cause bottlenecks
in other services such as websocket connection, API requests , etc.

Taking Python's multiprocessing library as an example, the basic usage would be:
* mp.Process to run all this code (and its indicators, signals, etc)
* mp.Array to update and read timestamp/price value.

In the Ticks data used (BNBUSDT), there will be a GAP on the first sell wave,
which is a good example of how the Renko Chart will look when this happens during a brick formation.
'''

import mplfinance as mpf
import pandas as pd
from matplotlib import animation

from renkodf import RenkoWS

df_ticks = pd.read_parquet('data/BNBUSDT-aggTrades-2023-06_9000Rows.parquet')

initial_timestamp = df_ticks['timestamp'].iat[0] # Timestamp (ms)
initial_price = df_ticks['close'].iat[0]

r = RenkoWS(initial_timestamp, initial_price, brick_size=0.04, ts_unit='ms')
initial_df = r.renko_df()

fig, axes = mpf.plot(initial_df, returnfig=True, volume=True,
                    figsize=(11, 8), panel_ratios=(2, 1),
                    title='\nBNBUSDT', type='candle', style='charles')
ax1 = axes[0]
ax2 = axes[2]

mpf.plot(initial_df,type='candle',ax=ax1,volume=ax2,axtitle='renko: wicks')

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

    df_wicks = r.renko_animate('wicks', max_len=100, keep=50)

    ax1.clear()
    ax2.clear()

    mpf.plot(df_wicks, type='candle', ax=ax1, volume=ax2, axtitle='renko: wicks')

ani = animation.FuncAnimation(fig, animate, interval=80)
mpf.show()