import unittest

import numpy as np
import pandas as pd
from renkodf import Renko, RenkoWS
from parameterized import parameterized

df_ticks_eurgbp = pd.read_parquet("../examples/data/EURGBP_T1_cT.parquet")
df_ticks_us30 = pd.read_parquet("../examples/data/US30_T1_cT.parquet")

eurgbp_len = len(df_ticks_eurgbp)
us30_len = len(df_ticks_us30)

for df_tick in [df_ticks_eurgbp, df_ticks_us30]:
    df_tick.rename(columns={'bid': 'close'}, inplace=True)
    df_tick['timestamp'] = pd.DatetimeIndex(df_tick.index).asi8 # Timestamp (us)

r1 = Renko(df_ticks_eurgbp, 0.0003)
r2 = Renko(df_ticks_us30, 5)

r1_ws = RenkoWS(df_ticks_eurgbp['timestamp'].iat[0], df_ticks_eurgbp['close'].iat[0], 0.0003)
r2_ws = RenkoWS(df_ticks_us30['timestamp'].iat[0], df_ticks_us30['close'].iat[0], 5)

eurgbp_tuple = (df_ticks_eurgbp['timestamp'].to_numpy(), df_ticks_eurgbp['close'].to_numpy())
for i in range(1, eurgbp_len):
    r1_ws.add_prices(eurgbp_tuple[0][i], eurgbp_tuple[1][i])

us30_tuple = (df_ticks_us30['timestamp'].to_numpy(), df_ticks_us30['close'].to_numpy())
for i in range(1, us30_len):
    r2_ws.add_prices(us30_tuple[0][i], us30_tuple[1][i])

class MyTestCase(unittest.TestCase):
    _MODE_dict = ['normal', 'wicks', 'nongap', 'reverse-wicks', 'reverse-nongap', 'fake-r-wicks', 'fake-r-nongap']

    @parameterized.expand(_MODE_dict)
    def test_eurgbp(self, mode):
        df_valid = r1.renko_df(mode)
        df = r1_ws.renko_df(mode) # without forming renko
        df.drop(df.head(2).index, inplace = True) # initial brick + first bar after.

        ohlcv = ['open', 'high', 'low', 'close', 'volume']
        integers = ['direction', 'is_reversal',]

        # "RenkoWS(real-time) OHLCV with floating-point-arithmetic should be strictly equal to Renko(backtest)"
        res = np.array_equal(df_valid[ohlcv + integers].values, df[ohlcv + integers].values)
        self.assertTrue(res)
        res_index = np.array_equal(df_valid.index.values, df.index.values)
        self.assertTrue(res_index)

    @parameterized.expand(_MODE_dict)
    def test_us30(self, mode):
        df_valid = r2.renko_df(mode)
        df = r2_ws.renko_df(mode) # without forming renko
        df.drop(df.head(2).index, inplace = True) # initial brick + first bar after.

        ohlcv = ['open', 'high', 'low', 'close', 'volume']
        integers = ['direction', 'is_reversal',]

        # "RenkoWS(real-time) OHLCV with floating-point-arithmetic should be strictly equal to Renko(backtest)"
        res = np.array_equal(df_valid[ohlcv + integers].values, df[ohlcv + integers].values)
        self.assertTrue(res)
        res_index = np.array_equal(df_valid.index.values, df.index.values)
        self.assertTrue(res_index)

if __name__ == '__main__':
    unittest.main()