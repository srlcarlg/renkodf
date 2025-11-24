import unittest

import numpy as np
import pandas as pd
from renkodf import Renko, RenkoWS
from parameterized import parameterized

df_ticks = pd.read_parquet("../examples/data/US30_T1_cT.parquet")
df_ticks.rename(columns={'bid': 'close'}, inplace=True)
df_ticks['timestamp'] = pd.DatetimeIndex(df_ticks.index).asi8 # Timestamp (us)

r1_full = Renko(df_ticks, 5)

df_GET = df_ticks.loc[(df_ticks.index <= '2023-06-28 13:50')]
df_ticks = df_ticks.loc[(df_ticks.index >= '2023-06-28 13:50')]
ticks_len = len(df_ticks)

r1 = Renko(df_GET, 5)
ext_df = r1.to_rws()

r1_ws = RenkoWS(external_df=ext_df, ts_unit='us')
start_len = len(r1_ws.renko_df())

us30_tuple = (df_ticks['timestamp'].to_numpy(), df_ticks['close'].to_numpy())
for i in range(1, ticks_len):
    r1_ws.add_prices(us30_tuple[0][i], us30_tuple[1][i])

class MyTestCase(unittest.TestCase):
    _MODE_dict = ['normal', 'wicks', 'nongap', 'reverse-wicks', 'reverse-nongap', 'fake-r-wicks', 'fake-r-nongap']

    @parameterized.expand(_MODE_dict)
    def test_us30(self, mode):
        df_valid = r1_full.renko_df(mode)
        df = r1_ws.renko_df(mode) # without forming renko

        ohlcv = ['open', 'high', 'low', 'close']
        integers = ['direction', 'is_reversal',]

        if mode in ['wicks', 'nongap']:
            df_valid_reset = df_valid.reset_index(drop=True)
            df_reset = df.reset_index(drop=True)

            open_diff = df_valid_reset['open'].compare(df_reset['open'])
            high_diff = df_valid_reset['high'].compare(df_reset['high'])
            volume_diff = df_valid_reset['volume'].compare(df_reset['volume'])

            if mode == 'nongap':
                self.assertEqual(1, len(open_diff))
                self.assertEqual(start_len, open_diff.index[0])
                ohlcv.remove('open')

            ohlcv.remove('high')
            # Just one diff
            self.assertEqual(1, len(high_diff))
            self.assertEqual(1, len(volume_diff))
            # This diff is the first-bar of RenkoWS after the last-bar from external_df
            self.assertEqual(start_len, high_diff.index[0])
            self.assertEqual(start_len, volume_diff.index[0])

        # "RenkoWS(real-time) OHLCV with floating-point-arithmetic should be strictly equal to Renko(backtest)"
        res = np.array_equal(df_valid[ohlcv + integers].values, df[ohlcv + integers].values)
        self.assertTrue(res)
        res_index = np.array_equal(df_valid.index.values, df.index.values)
        self.assertTrue(res_index)

if __name__ == '__main__':
    unittest.main()