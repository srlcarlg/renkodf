"""
renkodf
=====
Transform Tick Data into OHLCV Renko Dataframe!
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import mplfinance as mpf

_MODE_dict = ['normal', 'wicks', 'nongap', 'reverse-wicks', 'reverse-nongap', 'fake-r-wicks', 'fake-r-nongap']


class Renko:
    def __init__(self, df_ticks: pd.DataFrame, brick_size: float, divide_by: int = 2):
        """
        Create Renko OHLCV dataframe with existing Ticks data.

        Usage
        ------
        >> from renkodf import Renko \n
        >> r = Renko(df_ticks, brick_size) \n
        >> df = r.renkodf() \n

        Parameters
        ----------
        df_ticks : dataframe
            Only two columns are required:

            * "close": Mandatory.
            * "datetime": If is not present, the index will be used.
        brick_size : float
            The minimum/lowest brick size is (tick size * 2) => (0.00002 brick of 0.00001 tick size)
        divide_by : int
            The value to divide the 'len(df_ticks)' used as "fixed size" to create numpy arrays.
            Useful to reduce the peak RAM usage (pre-allocation).
            It's recommended to use values <= 10.
        """

        if brick_size is None or brick_size <= 0:
            raise ValueError("brick_size cannot be 'None' or '<= 0'")
        if divide_by is None or divide_by <= 0:
            raise ValueError("divide_by cannot be 'None' or '<= 0'")
        if 'datetime' not in df_ticks.columns:
            df_ticks["datetime"] = df_ticks.index
        if 'close' not in df_ticks.columns:
            raise ValueError("Column 'close' doesn't exist!")

        self._brick_size = brick_size

        # np.shares_memory() = True
        tick_prices = df_ticks['close'].to_numpy()
        tick_dates = df_ticks['datetime'].to_numpy()
        tick_len = len(tick_prices)

        renko_tuple = self._create_renko(tick_dates, tick_prices, tick_len, brick_size, divide_by)

        self._df_renko = pd.DataFrame(zip(*renko_tuple), columns=[
            'datetime', 'open', 'high', 'low', 'close', 'volume',
            'direction', 'is_reversal', 'tick_index_open', 'tick_index_close',
            'normal_high', 'normal_low', 'nongap_open',
            'reverse_nongap_open', 'reverse_fake_nongap_open',
            'reverse_high', 'reverse_low',
            'fake_high', 'fake_low'
        ])
        self._df_renko.index = pd.DatetimeIndex(self._df_renko["datetime"])
        self._df_renko.drop(self._df_renko['datetime'].iat[0], inplace=True)

    def _create_renko(self, tick_dates, tick_prices, tick_len, brick_size, divide_by):
        # Each variable will be a dataframe column
        # There's a massive speed up over dictionary { string key : list value }
        # np.array RAM usage is still lower/optimized than list "deepcopy([0.0] * half_len)"
        half_len = int(tick_len / divide_by)
        datetime = np.empty(half_len, dtype = np.ndarray)

        # ohlcv + utils
        open, high, low, close = (deepcopy(np.empty(half_len)) for _ in range(4))
        volume, direction, tick_index_open, tick_index_close, is_reversal = \
            (deepcopy(np.empty(half_len, dtype = np.int64)) for _ in range(5))

        # renko modes
        normal_high, normal_low, nongap_open, reverse_high, reverse_low, fake_high, fake_low = \
            (deepcopy(np.empty(half_len)) for _ in range(7))
        reverse_nongap_open, reverse_fake_nongap_open = (deepcopy(np.empty(half_len)) for _ in range(2))

        initial_price = (tick_prices[0] // brick_size) * brick_size

        # for loop
        wick_min_i, wick_max_i = initial_price, initial_price
        volume_i, tick_open_i, tick_close_i = 1, 1, 1

        # avoid [close/direction] arrays
        last_renko, last_direction = initial_price, 0

        # LLM helped here, division is slower than multiply
        invariant_brick = 1.0 / brick_size

        idx = 0
        for i in range(1, tick_len):
            price = tick_prices[i]

            wick_min_i = price if price < wick_min_i else wick_min_i
            wick_max_i = price if price > wick_max_i else wick_max_i
            volume_i += 1
            tick_close_i = i

            current_n_bricks = (price - last_renko) * invariant_brick
            if abs(current_n_bricks) < 1:
                continue

            is_up = current_n_bricks > 0
            current_direction = 1 if is_up else -1

            # Some micro-optimization here, it's better than / equivalent to:
            # (is_up and last_direction >= 0) or (not is_up and last_direction <= 0)
            is_same_direction = current_direction * last_direction >= 0

            total_same_bricks = current_n_bricks if is_same_direction else 0

            # >= 2 can be a 'GAP' or 'OPPOSITE DIRECTION'.
            # In both cases we add the current wick/volume to the first brick and 'reset' the value of both, since:
            # If it's a GAP:
            # - The following bricks after first brick will be 'artificial' since the price has 'skipped' that price region.
            # - (the reason of 'total_same_bricks')
            # If it's a OPPOSITE DIRECTION:
            # - Only the first brick will be kept. (the reason of '2' multiply)
            if not is_same_direction and abs(current_n_bricks) >= 2:
                renko_multiply = 2
                renko_price = last_renko + (current_direction * renko_multiply * brick_size)

                # wick mode
                open_price = renko_price - brick_size if is_up else renko_price + brick_size
                wick = wick_min_i if is_up else wick_max_i
                high_value = wick if not is_up else renko_price
                low_value = wick if is_up else renko_price

                normal_high_value = open_price if not is_up else renko_price
                normal_low_value = open_price if is_up else renko_price

                datetime[idx], open[idx], high[idx], low[idx], close[idx], volume[idx] = \
                    tick_dates[i], open_price, high_value, low_value, renko_price, volume_i

                direction[idx], tick_index_open[idx], tick_index_close[idx] = \
                    current_direction, tick_open_i, tick_close_i

                # normal/nongap
                normal_high[idx], normal_low[idx] = normal_high_value, normal_low_value
                nongap_value = wick if (
                    is_up and open_price > low_value or
                    not is_up and open_price < high_value
                ) else open_price
                nongap_open[idx] = nongap_value

                # reverse-normal/nongap
                is_reversal_loop = True
                is_reversal[idx] = int(is_reversal_loop)
                reverse_high[idx] = high_value if is_reversal_loop else normal_high_value
                reverse_low[idx] = low_value if is_reversal_loop else normal_low_value
                reverse_nongap_open[idx] = nongap_value if is_reversal_loop else open_price

                # fake-r-wicks/nongap
                fake_wick = last_renko
                fake_high[idx] = fake_wick if (is_reversal_loop and not is_up) else normal_high_value
                fake_low[idx] = fake_wick if (is_reversal_loop and is_up) else normal_low_value
                reverse_fake_nongap_open[idx] = fake_wick if is_reversal_loop else open_price

                # reset
                wick_min_i = open_price if is_reversal_loop else renko_price
                wick_max_i = open_price if is_reversal_loop else renko_price
                tick_open_i, tick_close_i = i, i
                volume_i = 1

                # performance
                last_direction = current_direction
                last_renko = renko_price

                # chart index
                idx += 1

                # set remaining bricks
                total_same_bricks = current_n_bricks - 2 * current_direction

            same_bricks = abs(int(total_same_bricks))
            if same_bricks < 1:
                continue

            # Add all bricks in the same direction
            for not_in_use in range(same_bricks):
                renko_multiply = 1

                # repeat code
                # only changes is_reversal_loop variable to False
                renko_price = last_renko + (current_direction * renko_multiply * brick_size)

                # wick mode
                open_price = renko_price - brick_size if is_up else renko_price + brick_size
                wick = wick_min_i if is_up else wick_max_i
                high_value = wick if not is_up else renko_price
                low_value = wick if is_up else renko_price

                normal_high_value = open_price if not is_up else renko_price
                normal_low_value = open_price if is_up else renko_price

                datetime[idx], open[idx], high[idx], low[idx], close[idx], volume[idx] = \
                    tick_dates[i], open_price, high_value, low_value, renko_price, volume_i

                direction[idx], tick_index_open[idx], tick_index_close[idx] = \
                    current_direction, tick_open_i, tick_close_i

                # normal/nongap
                normal_high[idx], normal_low[idx] = normal_high_value, normal_low_value
                nongap_value = wick if (
                    is_up and open_price > low_value or
                    not is_up and open_price < high_value
                ) else open_price
                nongap_open[idx] = nongap_value

                # reverse-normal/nongap
                is_reversal_loop = False
                is_reversal[idx] = int(is_reversal_loop)
                reverse_high[idx] = high_value if is_reversal_loop else normal_high_value
                reverse_low[idx] = low_value if is_reversal_loop else normal_low_value
                reverse_nongap_open[idx] = nongap_value if is_reversal_loop else open_price

                # fake-r-wicks/nongap
                fake_wick = last_renko
                fake_high[idx] = fake_wick if (is_reversal_loop and not is_up) else normal_high_value
                fake_low[idx] = fake_wick if (is_reversal_loop and is_up) else normal_low_value
                reverse_fake_nongap_open[idx] = fake_wick if is_reversal_loop else open_price

                # reset
                wick_min_i = open_price if is_reversal_loop else renko_price
                wick_max_i = open_price if is_reversal_loop else renko_price
                tick_open_i, tick_close_i = i, i
                volume_i = 1

                # performance
                last_direction = current_direction
                last_renko = renko_price

                # chart index
                idx += 1

        arrays = (datetime, open, high, low, close, volume,
                 direction, is_reversal, tick_index_open, tick_index_close,
                 normal_high, normal_low, nongap_open,
                 reverse_nongap_open, reverse_fake_nongap_open,
                 reverse_high, reverse_low,
                 fake_high, fake_low)
        return (arr[:idx] for arr in arrays)

    def plot(self, mode: str = "wicks", volume: bool = True, df: pd.DataFrame = None, add_plots: [] = None):
        """
        Redundant function only to plot the Renko Chart with fewer lines of code. \n
        If parameter "df" is used: the "add_plots" is mandatory. \n
        If parameter "df" is empty: only the 'renko_df' of the current instance will be plotted.

        Notes
        -----
        This function is equivalent to: \n
        > mpf.plot(df, type='candle', volume=True, style="charles") \n
        > mpf.plot(df, type='candle', volume=True, style="charles", addplot=[]) \n
        > mpf.show() \n

        Parameters
        ----------
        mode : str
            The method for building the renko dataframe, described in the function 'renko_df'.
        volume : bool
            Plot with Volume or not.
        df : dataframe
            External dataframe, usually with new columns for plotting indicators, signals, etc.
        add_plots : list
            A list with instances of mpf.make_addplot().
        """
        if df is not None and add_plots is None:
            raise ValueError("If 'df' parameter is used, 'add_plots' is mandatory!")

        if df is not None:
            mpf.plot(df, type='candle', style="charles", volume=volume, addplot=add_plots,
                     title=f"\n renko: {mode} \nbrick size: {self._brick_size}")
        else:
            df_renko = self.renko_df(mode)
            mpf.plot(df_renko, type='candle', style="charles", volume=volume,
                     title=f"\n renko: {mode} \nbrick size: {self._brick_size}")

        return mpf.show()

    def renko_df(self, mode: str = "wicks", utils_columns: bool = True):
        """
        Returns 'Renko OHLCV' dataframe by the given mode.

        Parameters
        ----------
        mode : str
            The method for building the renko dataframe, there are 7 modes available, where 3 are significant variations:

              * "normal" : Standard Renko.
              * "wicks" : Standard Renko with Wicks. (default)
              * "nongap": Same logic of 'wicks' mode but the OPEN will have the same value as the respective wick.
              * "reverse-wicks": 'wicks' only on price reversals.
              * "reverse-nongap": 'nongap' only in price reversals.
              * "fake-r-wicks": fake reverse wicks, where it will have the same value as the Previous Close.
              * "fake-r-nongap": fake reverse nongap, where it will have the same value as the Previous Close.

        utils_columns : bool
            Simple/Useful columns for backtesting or analysis:

                * 'direction' : 1 UP / -1 DOWN
                * 'is_reversal' : 0 False / 1 True
                * 'tick_index_open' : Self-explanatory
                * 'tick_index_close' : Self-explanatory

        Returns
        -------
        x : dataframe
            pandas.Dataframe
        """
        if mode not in _MODE_dict:
            raise ValueError(f"Only {_MODE_dict} options are valid.")

        df = self._df_renko.copy()
        df.drop(columns=['datetime'], inplace=True)

        # some helpers for repetitive code
        nongap_columns = ['nongap_open']
        normal_columns = ['normal_high', 'normal_low']
        reverse_columns = ['reverse_high', 'reverse_low']
        fake_r_columns = ['fake_high', 'fake_low']
        nongap_reverse_columns = ['reverse_nongap_open', 'reverse_fake_nongap_open']
        remaining_columns = ['direction', 'is_reversal', 'tick_index_open', 'tick_index_close']

        highlow_columns = reverse_columns + fake_r_columns + nongap_reverse_columns
        nn_columns = nongap_columns + normal_columns

        # drop non-related columns by mode
        drop_columns = {
            'normal': highlow_columns + nongap_columns,
            'wicks': highlow_columns + nn_columns,
            'nongap': highlow_columns + normal_columns,
            'reverse-wicks': fake_r_columns + nn_columns,
            'reverse-nongap': fake_r_columns + nn_columns + [nongap_reverse_columns[1]],
            'fake-r-wicks': reverse_columns + nn_columns,
            'fake-r-nongap': reverse_columns + nn_columns + [nongap_reverse_columns[0]]
        }
        to_drop = drop_columns[mode]
        df.drop(columns=to_drop, inplace=True)

        # utils
        if not utils_columns:
            df.drop(columns=remaining_columns, inplace=True)

        # 'wicks' = default
        if mode == 'wicks':
            return df

        # drop "wicks" 'high/low' columns, also 'open' if needed
        to_replace = ['high', 'low'] if mode != 'nongap' else ['open']
        if mode in ['reverse-nongap', 'fake-r-nongap']:
            to_replace += ['open']
        df.drop(columns=to_replace, inplace=True)

        # rename specific renko-mode columns to 'open/high/low'
        mode_columns = {
            'normal': normal_columns,
            'nongap': nongap_columns,
            'reverse-wicks': reverse_columns,
            'reverse-nongap': reverse_columns + [nongap_reverse_columns[0]],
            'fake-r-wicks': fake_r_columns,
            'fake-r-nongap': fake_r_columns + [nongap_reverse_columns[1]],
        }
        keys = mode_columns[mode]
        values = to_replace
        if mode in ['reverse-nongap', 'fake-r-nongap']:
            values += ['open']

        to_rename = dict(zip(keys, values))
        df.rename(columns=to_rename, inplace=True)

        # re-order columns
        order = ['open', 'high', 'low', 'close', 'volume']
        if utils_columns:
            order += remaining_columns

        return df[order]

    def to_rws(self, use_iloc: int = None):
        """
        Returns 'Renko OHLCV' dataframe with all renko modes,
        which can be used as initial data in the 'RenkoWS' class. \n

        The DatetimeIndex will be converted to Timestamp (dynamic unit)

        Parameters
        ----------
        use_iloc : int
            * If positive: First nº rows will be returned
            * If negative: Last nº rows will be returned

        Returns
        -------
        x : dataframe
            pandas.Dataframe
        """
        df = self._df_renko.copy()
        df.drop(columns=['tick_index_open', 'tick_index_close'], inplace=True)

        df['brick_size'] = self._brick_size
        df['timestamp'] = pd.DatetimeIndex(df["datetime"]).asi8 # Timestamp with dynamic unit [D, s, ms, us, ns]
        df.drop(columns=['datetime'], inplace=True)

        if use_iloc is not None:
            if use_iloc < 0:
                return df.iloc[use_iloc:]
            else:
                return df.iloc[:use_iloc]
        else:
            return df


class RenkoWS:
    def __init__(self, ws_timestamp: int = None, ws_price: float = None,
                 brick_size: float = None, external_df: pd.DataFrame = None,
                 ts_unit: str = 'us'):
        """
        Create real-time Renko charts, usually over a WebSocket connection.

        Usage
        -----
        >> from renkodf import RenkoWS \n
        >> r = RenkoWS(your combination) \n
        >> # At every price change \n
        >> r.add_prices(ws_timestamp, ws_price) \n
        >> df = r.renko_animate() \n

        Notes
        -----
        Only the following combinations are possible: \n
        > RenkoWS(ws_timestamp, ws_price, brick_size, ts_unit) \n
        > RenkoWS(external_df, ts_unit) \n

        Parameters
        ----------
        ws_timestamp : int
            Self-explanatory.
        ws_price : float
            Self-explanatory.
        brick_size : float
            Cannot be less than or equal to 0.00...
        external_df : dataframe
            The dataframe made from Renko.to_rws()
        ts_unit : str
            The expected 'ws_timestamp' unit to be used at pandas.to_datetime() => [D, s, ms, us, ns]
        """
        if external_df is None:
            if brick_size is None or brick_size <= 0:
                raise ValueError("brick_size cannot be 'None' or '<= 0'")
            if ws_price is None:
                raise ValueError("ws_price cannot be 'None'")
            if ws_timestamp is None:
                raise ValueError("ws_timestamp cannot be 'None'")

        self._brick_size = brick_size if external_df is None else external_df['brick_size'].iat[0]
        self._ts_unit = ts_unit

        if external_df is None:
            initial_price = (ws_price // brick_size) * brick_size
            init_value = np.array([initial_price])
            init_int = np.array([1], dtype=np.int64)

            # ohlcv + utils
            timestamp = np.array([ws_timestamp], dtype=np.int64)
            open, high, low, close = (deepcopy(init_value) for _ in range(4))
            volume, direction, is_reversal = (deepcopy(init_int) for _ in range(3))
            # renko modes
            normal_high, normal_low, nongap_open, reverse_high, reverse_low, fake_high, fake_low = \
                (deepcopy(init_value) for _ in range(7))
            reverse_nongap_open, reverse_fake_nongap_open = (deepcopy(init_value) for _ in range(2))

            renko_tuple = (timestamp, open, high, low, close, volume,
                           direction, is_reversal,
                           normal_high, normal_low, nongap_open,
                           reverse_nongap_open, reverse_fake_nongap_open,
                           reverse_high, reverse_low,
                           fake_high, fake_low)

            self._df_renko = pd.DataFrame(zip(*renko_tuple), columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'direction', 'is_reversal',
                'normal_high', 'normal_low', 'nongap_open',
                'reverse_nongap_open', 'reverse_fake_nongap_open',
                'reverse_high', 'reverse_low',
                'fake_high', 'fake_low'
            ])

            self._df_renko.index = pd.DatetimeIndex(pd.to_datetime(self._df_renko["timestamp"], unit=ts_unit))
            self._df_renko.index.name = 'datetime'
        else:
            external_df.drop(columns=['brick_size'], inplace=True)
            self._df_renko = external_df

        self._initial_df = self._df_renko.copy()
        self._invariant_brick = 1.0 / self._brick_size
        self._volume_i = self._df_renko['volume'].iat[-1]
        self._wick_min_i, self._wick_max_i, self._last_renko = (self._df_renko['close'].iat[-1] for _ in range(3))
        self._last_direction = self._df_renko['direction'].iat[-1]

        self._ws_timestamp = ws_timestamp
        self._ws_price = ws_price

    def add_prices(self, ws_timestamp: int, ws_price: float, gap_tolerance: int = 200):
        """
        Determine if new renko-bars should be added according to the current price.

        Must be called at every price change.

        Parameters
        ----------
        ws_timestamp : int
            Self-explanatory.
        ws_price : float
            Self-explanatory.
        gap_tolerance : int
            It's the "fixed size" value used to create numpy arrays.

            It also means that when a 'gap' occurs, the maximum 'artificial' bricks count should be <= gap_tolerance value.
        """
        self._ws_timestamp = ws_timestamp
        self._ws_price = ws_price

        self._wick_min_i = ws_price if ws_price < self._wick_min_i else self._wick_min_i
        self._wick_max_i = ws_price if ws_price > self._wick_max_i else self._wick_max_i
        self._volume_i += 1

        current_n_bricks = (ws_price - self._last_renko) * self._invariant_brick
        if abs(current_n_bricks) < 1:
            return

        # +200 => tolerance of 200 new bricks at gap
        half_len = gap_tolerance

        # ohlcv + utils
        open, high, low, close = (deepcopy(np.empty(half_len)) for _ in range(4))
        timestamp, volume, direction, is_reversal = (deepcopy(np.empty(half_len, dtype=np.int64)) for _ in range(4))
        # renko modes
        normal_high, normal_low, nongap_open, reverse_high, reverse_low, fake_high, fake_low = \
            (deepcopy(np.empty(half_len)) for _ in range(7))
        reverse_nongap_open, reverse_fake_nongap_open = (deepcopy(np.empty(half_len)) for _ in range(2))

        brick_size = self._brick_size
        idx = 0

        is_up = current_n_bricks > 0
        current_direction = 1 if is_up else -1
        is_same_direction = current_direction * self._last_direction >= 0
        total_same_bricks = current_n_bricks if is_same_direction else 0

        if not is_same_direction and abs(current_n_bricks) >= 2:
            renko_multiply = 2
            renko_price = self._last_renko + (current_direction * renko_multiply * brick_size)

            # wick mode
            open_price = renko_price - brick_size if is_up else renko_price + brick_size
            wick = self._wick_min_i if is_up else self._wick_max_i
            high_value = wick if not is_up else renko_price
            low_value = wick if is_up else renko_price

            normal_high_value = open_price if not is_up else renko_price
            normal_low_value = open_price if is_up else renko_price

            timestamp[idx], open[idx], high[idx], low[idx], close[idx], volume[idx] = \
                ws_timestamp, open_price, high_value, low_value, renko_price, self._volume_i
            direction[idx] = current_direction

            # normal/nongap
            normal_high[idx], normal_low[idx] = normal_high_value, normal_low_value
            nongap_value = wick if (
                is_up and open_price > low_value or
                not is_up and open_price < high_value
            ) else open_price
            nongap_open[idx] = nongap_value

            # reverse-normal/nongap
            is_reversal_loop = True
            is_reversal[idx] = int(is_reversal_loop)
            reverse_high[idx] = high_value if is_reversal_loop else normal_high_value
            reverse_low[idx] = low_value if is_reversal_loop else normal_low_value
            reverse_nongap_open[idx] = nongap_value if is_reversal_loop else open_price

            # fake-r-wicks/nongap
            fake_wick = self._last_renko
            fake_high[idx] = fake_wick if (is_reversal_loop and not is_up) else normal_high_value
            fake_low[idx] = fake_wick if (is_reversal_loop and is_up) else normal_low_value
            reverse_fake_nongap_open[idx] = fake_wick if is_reversal_loop else open_price

            # reset
            self._wick_min_i = open_price if is_reversal_loop else renko_price
            self._wick_max_i = open_price if is_reversal_loop else renko_price
            self._volume_i = 1

            # performance
            self._last_direction = current_direction
            self._last_renko = renko_price

            # chart index
            idx += 1

            # set remaining bricks
            total_same_bricks = current_n_bricks - 2 * current_direction

        same_bricks = abs(int(total_same_bricks))
        if same_bricks < 1 and abs(current_n_bricks) < 2:
            return

        # Add all bricks in the same direction
        for not_in_use in range(same_bricks):
            renko_multiply = 1

            # repeat code
            # only changes is_reversal_loop variable to False
            renko_price = self._last_renko + (current_direction * renko_multiply * brick_size)

            # wick mode
            open_price = renko_price - brick_size if is_up else renko_price + brick_size
            wick = self._wick_min_i if is_up else self._wick_max_i
            high_value = wick if not is_up else renko_price
            low_value = wick if is_up else renko_price

            normal_high_value = open_price if not is_up else renko_price
            normal_low_value = open_price if is_up else renko_price

            timestamp[idx], open[idx], high[idx], low[idx], close[idx], volume[idx] = \
                ws_timestamp, open_price, high_value, low_value, renko_price, self._volume_i
            direction[idx] = current_direction

            # normal/nongap
            normal_high[idx], normal_low[idx] = normal_high_value, normal_low_value
            nongap_value = wick if (
                is_up and open_price > low_value or
                not is_up and open_price < high_value
            ) else open_price
            nongap_open[idx] = nongap_value

            # reverse-normal/nongap
            is_reversal_loop = False
            is_reversal[idx] = int(is_reversal_loop)
            reverse_high[idx] = high_value if is_reversal_loop else normal_high_value
            reverse_low[idx] = low_value if is_reversal_loop else normal_low_value
            reverse_nongap_open[idx] = nongap_value if is_reversal_loop else open_price

            # fake-r-wicks/nongap
            fake_wick = self._last_renko
            fake_high[idx] = fake_wick if (is_reversal_loop and not is_up) else normal_high_value
            fake_low[idx] = fake_wick if (is_reversal_loop and is_up) else normal_low_value
            reverse_fake_nongap_open[idx] = fake_wick if is_reversal_loop else open_price

            # reset
            self._wick_min_i = open_price if is_reversal_loop else renko_price
            self._wick_max_i = open_price if is_reversal_loop else renko_price
            self._volume_i = 1

            # performance
            self._last_direction = current_direction
            self._last_renko = renko_price

            # chart index
            idx += 1

        arrays = (timestamp, open, high, low, close, volume,
                 direction, is_reversal,
                 normal_high, normal_low, nongap_open,
                 reverse_nongap_open, reverse_fake_nongap_open,
                 reverse_high, reverse_low,
                 fake_high, fake_low)
        renko_tuple = (arr[:idx] for arr in arrays)

        brick_df = pd.DataFrame(zip(*renko_tuple), columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'direction', 'is_reversal',
            'normal_high', 'normal_low', 'nongap_open',
            'reverse_nongap_open', 'reverse_fake_nongap_open',
            'reverse_high', 'reverse_low',
            'fake_high', 'fake_low'
        ])
        brick_df.index = pd.DatetimeIndex(pd.to_datetime(brick_df["timestamp"], unit=self._ts_unit))
        brick_df.index.name = 'datetime'

        self._df_renko = pd.concat([self._df_renko, brick_df])

    def renko_df(self, mode: str = "wicks", utils_columns: bool = True):
        """
        Returns 'Renko OHLCV' dataframe by the given mode.

        Can be called after 'add_prices(ws_timestamp, ws_price)'.

        However, only completed bricks will be shown.
        """
        if mode not in _MODE_dict:
            raise ValueError(f"Only {_MODE_dict} options are valid.")

        df = self._df_renko.copy()

        # some helpers for repetitive code
        nongap_columns = ['nongap_open']
        normal_columns = ['normal_high', 'normal_low']
        reverse_columns = ['reverse_high', 'reverse_low']
        fake_r_columns = ['fake_high', 'fake_low']
        nongap_reverse_columns = ['reverse_nongap_open', 'reverse_fake_nongap_open']
        remaining_columns = ['direction', 'is_reversal']

        highlow_columns = reverse_columns + fake_r_columns + nongap_reverse_columns
        nn_columns = nongap_columns + normal_columns

        # drop non-related columns by mode
        drop_columns = {
            'normal': highlow_columns + nongap_columns,
            'wicks': highlow_columns + nn_columns,
            'nongap': highlow_columns + normal_columns,
            'reverse-wicks': fake_r_columns + nn_columns,
            'reverse-nongap': fake_r_columns + nn_columns + [nongap_reverse_columns[1]],
            'fake-r-wicks': reverse_columns + nn_columns,
            'fake-r-nongap': reverse_columns + nn_columns + [nongap_reverse_columns[0]]
        }
        to_drop = drop_columns[mode]
        df.drop(columns=to_drop, inplace=True)

        # utils
        if not utils_columns:
            df.drop(columns=remaining_columns, inplace=True)

        # 'wicks' = default
        if mode == 'wicks':
            return df

        # drop "wicks" 'high/low' columns, also 'open' if needed
        to_replace = ['high', 'low'] if mode != 'nongap' else ['open']
        if mode in ['reverse-nongap', 'fake-r-nongap']:
            to_replace += ['open']
        df.drop(columns=to_replace, inplace=True)

        # rename specific renko-mode columns to 'open/high/low'
        mode_columns = {
            'normal': normal_columns,
            'nongap': nongap_columns,
            'reverse-wicks': reverse_columns,
            'reverse-nongap': reverse_columns + [nongap_reverse_columns[0]],
            'fake-r-wicks': fake_r_columns,
            'fake-r-nongap': fake_r_columns + [nongap_reverse_columns[1]],
        }
        keys = mode_columns[mode]
        values = to_replace
        if mode in ['reverse-nongap', 'fake-r-nongap']:
            values += ['open']

        to_rename = dict(zip(keys, values))
        df.rename(columns=to_rename, inplace=True)

        # re-order columns
        order = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if utils_columns:
            order += remaining_columns

        return df[order]

    def renko_animate(self, mode: str = 'wicks', max_len: int = 500, keep: int = 250):
        """
        Should be called after 'add_prices(ws_timestamp, ws_price)'.

        Alternatively, call 'renko_df()' if the forming renko is not required.

        Parameters
        ----------
        mode : str
            The method for building the renko dataframe, described in the Renko.renko_df().
        max_len : int
            Once reached, the 'Renko OHLCV' values will be deleted.

            If max_len == 0, this behavior will not be triggered.
        keep : int
            Keep last nº values after deletion.

        Returns
        -------
        x : dataframe
            pandas.Dataframe
        """
        df = self.renko_df(mode)
        df_length = len(df)

        ws_timestamp = self._ws_timestamp
        ws_price = self._ws_price

        raw_ws = {
            "timestamp": [ws_timestamp],
            "open": [ws_price],
            "high": [ws_price],
            "low": [ws_price],
            "close": [ws_price],
            "volume": self._volume_i,
            "direction": [0],
            "is_reversal": [0]
        }
        if df_length < 1:
            raw_ws["open"][-1] = self._initial_df["close"].iat[-1]
            raw_ws["high"][-1] = self._wick_max_i
            raw_ws["low"][-1] = self._wick_min_i

            df_ws = pd.DataFrame(raw_ws)
            df_ws.index = pd.DatetimeIndex(pd.to_datetime(df_ws["timestamp"], unit=self._ts_unit))
            df_ws.index.name = 'datetime'
            df_ws.drop(columns=['timestamp'], inplace=True)

            return pd.concat([self._initial_df, df_ws])

        # Forming wick
        raw_ws["high"][-1] = self._wick_max_i if mode != 'normal' else ws_price
        raw_ws["low"][-1] = self._wick_min_i if mode != 'normal' else ws_price

        nongap_rule = mode in ['nongap', 'reverse-nongap', 'fake-r-nongap']
        last_renko_close = df["close"].iat[-1]
        last_renko_open = df["open"].iat[-1]
        # Last Renko (UP)
        if last_renko_close > last_renko_open:
            if ws_price > last_renko_close:
                raw_ws["open"][-1] = self._wick_min_i if nongap_rule else last_renko_close
                if mode == "normal":
                    raw_ws["low"][-1] = last_renko_close
            else:
                if ws_price < last_renko_open:
                    raw_ws["open"][-1] = self._wick_max_i if nongap_rule else last_renko_open
                    if mode == "normal":
                        raw_ws["high"][-1] = last_renko_open
        # Last Renko (DOWN)
        else:
            if ws_price < last_renko_close:
                raw_ws["open"][-1] = self._wick_max_i if nongap_rule else last_renko_close
                if mode == "normal":
                    raw_ws["high"][-1] = last_renko_close
            else:
                if ws_price > last_renko_open:
                    raw_ws["open"][-1] = self._wick_min_i if nongap_rule else last_renko_open
                    if mode == "normal":
                        raw_ws["low"][-1] = last_renko_open

        is_up = raw_ws['close'][-1] > raw_ws['open'][-1]
        is_down = raw_ws['close'][-1] < raw_ws['open'][-1]
        raw_ws['direction'][-1] = 1 if is_up else -1 if is_down else 0

        df_ws = pd.DataFrame(raw_ws)
        df_ws.index = pd.DatetimeIndex(pd.to_datetime(df_ws["timestamp"], unit=self._ts_unit))
        df_ws.index.name = 'datetime'

        if max_len != 0 and df_length >= max_len:
            self._df_renko.drop(self._df_renko.index[:(max_len - keep)], inplace=True)

        return pd.concat([df, df_ws])
