'''
This file contains a simple animation demo using mplfinance.

Imagine that we make an API request where:
  * we receive the latest asset data; up to the present moment,
  * create the Renko Chart from it,
  * and save it into a file.

Then we make a websocket connection to this same asset and:
  * use the previous Renko Chart as starting point, without creating the chart from scratch!

Useful when you need large initial data to calculate/plot your technical indicators.
'''

import mplfinance as mpf
import pandas as pd
from matplotlib import animation

from renkodf import Renko, RenkoWS

df_ticks = pd.read_parquet('data/US30_T1_cT.parquet')
df_ticks.rename(columns={'bid': 'close'}, inplace=True)
df_ticks['timestamp'] = pd.DatetimeIndex(df_ticks.index).asi8 # Timestamp (us)

df_GET = df_ticks.loc[(df_ticks.index <= '2023-06-28 13:50')]
df_ticks = df_ticks.loc[(df_ticks.index >= '2023-06-28 13:50')]

# API request
r = Renko(df_GET, brick_size=5)
ext_df = r.to_rws()  # Save this

# Load the file and chosen its Renko Mode
r = RenkoWS(external_df=ext_df, ts_unit='us')
initial_df = r.renko_df()

fig, axes = mpf.plot(initial_df, returnfig=True, volume=True,
                    figsize=(11, 8), panel_ratios=(2, 1),
                    title='\nUS30', type='candle', style='yahoo')
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

    df_wicks = r.renko_animate('wicks', max_len=1000, keep=500)

    ax1.clear()
    ax2.clear()

    mpf.plot(df_wicks, type='candle', ax=ax1, volume=ax2, axtitle='renko: wicks')


ani = animation.FuncAnimation(fig, animate, interval=80)
mpf.show()