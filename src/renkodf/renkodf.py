"""
renkodf
=====
Transform Tick Data into OHLCV Renko Dataframe!
"""
import gc
import numpy as np
import pandas as pd
import numba
import mplfinance as mpf

_MODE_dict = ['normal', 'wicks', 'nongap', 'reverse-wicks', 'reverse-nongap', 'fake-r-wicks', 'fake-r-nongap']

@numba.jit(nopython=True)
def _process_prices(prices, brick_size, df_len):
    initial_price = (prices[0] // brick_size) * brick_size
    
    rsd_price = np.empty(df_len * 2, dtype=np.float64)
    rsd_direction = np.empty(df_len * 2, dtype=np.float64)
    rsd_wick = np.empty(df_len * 2, dtype=np.float64)
    rsd_volume = np.empty(df_len * 2, dtype=np.int64)
    rsd_index = np.empty(df_len * 2, dtype=np.int64)
    
    rsd_price[0] = initial_price
    rsd_direction[0] = 0
    rsd_wick[0] = initial_price
    rsd_volume[0] = 1
    rsd_index[0] = 0
    rsd_count = 1
    
    wick_min = initial_price
    wick_max = initial_price
    volume = 1
    
    for i in range(1, df_len):
        price = prices[i]
        wick_min = min(price, wick_min)
        wick_max = max(price, wick_max)
        volume += 1
        
        last_price = rsd_price[rsd_count - 1]
        current_n_bricks = (price - last_price) / brick_size
        current_direction = np.sign(current_n_bricks)
        
        if current_direction == 0:
            continue
            
        last_direction = rsd_direction[rsd_count - 1]
        is_same_direction = ((current_direction > 0 and last_direction >= 0) or 
                           (current_direction < 0 and last_direction <= 0))
        
        total_same_bricks = current_n_bricks if is_same_direction else 0
        
        if not is_same_direction and abs(current_n_bricks) >= 2:
            renko_price = last_price + (current_direction * 2 * brick_size)
            wick = wick_min if current_n_bricks > 0 else wick_max
            
            rsd_price[rsd_count] = renko_price
            rsd_direction[rsd_count] = current_direction
            rsd_wick[rsd_count] = wick
            rsd_volume[rsd_count] = volume
            rsd_index[rsd_count] = i
            rsd_count += 1
            
            volume = 1
            wick_min = renko_price if current_n_bricks > 0 else wick_min
            wick_max = renko_price if current_n_bricks < 0 else wick_max
            
            total_same_bricks = current_n_bricks - 2 * current_direction
        
        n_bricks = abs(int(total_same_bricks))
        for _ in range(n_bricks):
            renko_price = rsd_price[rsd_count - 1] + (current_direction * brick_size)
            wick = wick_min if current_n_bricks > 0 else wick_max
            
            rsd_price[rsd_count] = renko_price
            rsd_direction[rsd_count] = current_direction
            rsd_wick[rsd_count] = wick
            rsd_volume[rsd_count] = volume
            rsd_index[rsd_count] = i
            rsd_count += 1
            
            volume = 1
            wick_min = renko_price if current_n_bricks > 0 else wick_min
            wick_max = renko_price if current_n_bricks < 0 else wick_max
    
    return (rsd_price[:rsd_count], rsd_direction[:rsd_count], 
            rsd_wick[:rsd_count], rsd_volume[:rsd_count], rsd_index[:rsd_count])

@numba.jit(nopython=True)
def _build_ohlc_wicks(prices, directions, wicks, brick_size):
    n = len(prices)
    opens = np.empty(n, dtype=np.float64)
    highs = np.empty(n, dtype=np.float64)
    lows = np.empty(n, dtype=np.float64)
    closes = np.empty(n, dtype=np.float64)
    
    prev_direction = 0
    prev_close = 0
    prev_close_up = 0
    prev_close_down = 0
    
    valid_count = 0
    for i in range(n):
        if directions[i] == 0:
            continue
            
        closes[valid_count] = prices[i]
        
        if directions[i] == 1.0:
            highs[valid_count] = prices[i]
            if prev_direction == 1:
                opens[valid_count] = prev_close_up
                lows[valid_count] = wicks[i]
            else:
                opens[valid_count] = prev_close + brick_size
                lows[valid_count] = wicks[i]
            prev_close_up = prices[i]
        elif directions[i] == -1.0:
            lows[valid_count] = prices[i]
            if prev_direction == -1:
                opens[valid_count] = prev_close_down
                highs[valid_count] = wicks[i]
            else:
                opens[valid_count] = prev_close - brick_size
                highs[valid_count] = wicks[i]
            prev_close_down = prices[i]
        
        prev_direction = directions[i]
        prev_close = prices[i]
        valid_count += 1
    
    return opens[2:valid_count], highs[2:valid_count], lows[2:valid_count], closes[2:valid_count]

class Renko:
    def __init__(self, df: pd.DataFrame, brick_size: float, add_columns: list = None, show_progress: bool = False):
        if brick_size is None or brick_size <= 0:
            raise ValueError("brick_size cannot be 'None' or '<= 0'")
        if 'datetime' not in df.columns:
            df["datetime"] = df.index
        if 'close' not in df.columns:
            raise ValueError("Column 'close' doesn't exist!")
        if add_columns is not None:
            if not set(add_columns).issubset(df.columns):
                raise ValueError(f"One or more of {add_columns} columns don't exist!")

        self._brick_size = brick_size
        self._custom_columns = add_columns
        self._df_len = len(df["close"])
        self._show_progress = show_progress
        self._df = df
        
        prices = df["close"].values.astype(np.float64)
        rsd_data = _process_prices(prices, brick_size, self._df_len)
        
        self._rsd = {
            "origin_index": rsd_data[4],
            "date": df["datetime"].values[rsd_data[4]],
            "price": rsd_data[0],
            "direction": rsd_data[1],
            "wick": rsd_data[2],
            "volume": rsd_data[3],
        }
        
        if add_columns is not None:
            for name in add_columns:
                self._rsd[name] = df[name].values[rsd_data[4]]

    def plot(self, mode: str = "wicks", volume: bool = True, df: pd.DataFrame = None, add_plots: [] = None):
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

    def renko_df(self, mode: str = "wicks"):
        if mode not in _MODE_dict:
            raise ValueError(f"Only {_MODE_dict} options are valid.")

        if mode == "wicks":
            opens, highs, lows, closes = _build_ohlc_wicks(
                self._rsd["price"], self._rsd["direction"], 
                self._rsd["wick"], self._brick_size
            )
            
            valid_idx = self._rsd["direction"] != 0
            valid_idx[0] = False
            valid_idx[1] = False
            
            valid_dates = self._rsd["date"][valid_idx]
            valid_volumes = self._rsd["volume"][valid_idx]
            
            # Ensure all arrays have same length
            min_len = min(len(opens), len(valid_dates[2:]), len(valid_volumes[2:]))
            
            df_dict = {
                "datetime": valid_dates[2:2+min_len],
                "open": opens[:min_len],
                "high": highs[:min_len],
                "low": lows[:min_len],
                "close": closes[:min_len],
                "volume": valid_volumes[2:2+min_len],
            }
            
            if self._custom_columns is not None:
                for name in self._custom_columns:
                    df_dict[name] = self._rsd[name][valid_idx][2:2+min_len]
        else:
            dates = self._rsd["date"]
            prices = self._rsd["price"]
            directions = self._rsd["direction"]
            wicks = self._rsd["wick"]
            volumes = self._rsd["volume"]
            indexes = list(range(len(prices)))
            brick_size = self._brick_size

            df_dict = {
                "datetime": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
            if self._custom_columns is not None:
                for name in self._custom_columns:
                    df_dict.update({name: []})

            reverse_rule = mode in ["normal", "wicks", "reverse-wicks", "fake-r-wicks"]
            fake_reverse_rule = mode in ["fake-r-nongap", "fake-r-wicks"]
            same_direction_rule = mode in ["wicks", "nongap"]

            prev_direction = 0
            prev_close = 0
            prev_close_up = 0
            prev_close_down = 0
            for price, direction, date, wick, volume, index in zip(prices, directions, dates, wicks, volumes, indexes):
                if direction != 0:
                    df_dict["datetime"].append(date)
                    df_dict["close"].append(price)
                    df_dict["volume"].append(volume)

                if direction == 1.0:
                    df_dict["high"].append(price)
                    if self._custom_columns is not None:
                        for name in self._custom_columns:
                            df_dict[name].append(self._rsd[name][index])
                    if prev_direction == 1:
                        df_dict["open"].append(wick if mode == "nongap" else prev_close_up)
                        df_dict["low"].append(wick if same_direction_rule else prev_close_up)
                    else:
                        if reverse_rule:
                            df_dict["open"].append(prev_close + brick_size)
                        elif mode == "fake-r-nongap":
                            df_dict["open"].append(prev_close_down)
                        else:
                            df_dict["open"].append(wick)

                        if mode == "normal":
                            df_dict["low"].append(prev_close + brick_size)
                        elif fake_reverse_rule:
                            df_dict["low"].append(prev_close_down)
                        else:
                            df_dict["low"].append(wick)
                    prev_close_up = price
                elif direction == -1.0:
                    df_dict["low"].append(price)
                    if self._custom_columns is not None:
                        for name in self._custom_columns:
                            df_dict[name].append(self._rsd[name][index])
                    if prev_direction == -1:
                        df_dict["open"].append(wick if mode == "nongap" else prev_close_down)
                        df_dict["high"].append(wick if same_direction_rule else prev_close_down)
                    else:
                        if reverse_rule:
                            df_dict["open"].append(prev_close - brick_size)
                        elif mode == "fake-r-nongap":
                            df_dict["open"].append(prev_close_up)
                        else:
                            df_dict["open"].append(wick)

                        if mode == "normal":
                            df_dict["high"].append(prev_close - brick_size)
                        elif fake_reverse_rule:
                            df_dict["high"].append(prev_close_up)
                        else:
                            df_dict["high"].append(wick)
                    prev_close_down = price
                else:
                    df_dict["datetime"].append(np.nan)
                    df_dict["low"].append(np.nan)
                    df_dict["close"].append(np.nan)
                    df_dict["high"].append(np.nan)
                    df_dict["open"].append(np.nan)
                    df_dict["volume"].append(np.nan)
                    if self._custom_columns is not None:
                        for name in self._custom_columns:
                            df_dict[name].append(np.nan)

                prev_direction = direction
                prev_close = price

            df_dict = {k: v[2:] for k, v in df_dict.items() if len(v) > 2}

        df = pd.DataFrame(df_dict)
        df.index = pd.DatetimeIndex(df["datetime"])
        df.drop(columns=['datetime'], inplace=True)

        return df

    def to_rws(self, use_iloc: int = None):
        rws = pd.DataFrame(self._rsd)
        rws.index = pd.DatetimeIndex(rws["date"]).asi8 // 10 ** 6
        rws.index.name = 'timestamp'
        rws['timestamp'] = rws.index
        rws['brick_size'] = self._brick_size

        if use_iloc is not None:
            if use_iloc < 0:
                return rws.iloc[use_iloc:]
            else:
                return rws.iloc[:use_iloc]
        else:
            return rws

class RenkoWS:
    def __init__(self, ws_timestamp: int = None, ws_price: float = None,
                 brick_size: float = None,
                 external_df: pd.DataFrame = None, external_mode: str = 'wicks'):
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
        > RenkoWS(ws_timestamp, ws_price, brick_size) \n
        > RenkoWS(external_df, external_mode) \n

        Parameters
        ----------
        ws_timestamp : int
            Timestamp in milliseconds.
        ws_price : float
            Self-explanatory.
        brick_size : float
            Cannot be less than or equal to 0.00...
        external_df : dataframe
            The dataframe made from Renko.to_rws()
        external_mode : str
            The method for building the external renko df, described in the 'Renko.renko_df()'.
        """
        if external_df is None:
            if brick_size is None or brick_size <= 0:
                raise ValueError("brick_size cannot be 'None' or '<= 0'")
            if ws_price is None:
                raise ValueError("ws_price cannot be 'None'")
            if ws_timestamp is None:
                raise ValueError("ws_timestamp cannot be 'None'")

        self._brick_size = brick_size if external_df is None else external_df['brick_size'].iat[0]

        initial_price = 0.0
        if external_df is None:
            initial_price = (ws_price // self._brick_size) * self._brick_size
            self._rsd = {
                "timestamp": [ws_timestamp],
                "price": [initial_price],
                "direction": [0],
                "wick": [initial_price],
                "volume": [1],
            }
        else:
            self._rsd = {
                "timestamp": external_df['timestamp'].to_list(),
                "price": external_df['price'].to_list(),
                "direction": external_df['direction'].to_list(),
                "wick": external_df['wick'].to_list(),
                "volume": external_df['volume'].to_list(),
            }

        if external_df is None:
            initial_df = {
                "timestamp": [ws_timestamp],
                "open": [initial_price],
                "high": [initial_price],
                "low": [initial_price],
                "close": [initial_price],
                "volume": [1]
            }
            initial_df = pd.DataFrame(initial_df, columns=["timestamp", "open", "high", "low", "close", "volume"])
            initial_df.index = pd.DatetimeIndex(
                pd.to_datetime(initial_df["timestamp"].values.astype(np.int64), unit="ms"))
            initial_df.drop(columns=['timestamp'], inplace=True)
        else:
            initial_df = self._renko_df(external_mode)

        self.initial_df = initial_df

        # For loop
        self._volume_i = 1
        self._wick_min_i = initial_price if external_df is None else external_df['price'].iat[-1]
        self._wick_max_i = initial_price if external_df is None else external_df['price'].iat[-1]

        self._ws_timestamp = ws_timestamp
        self._ws_price = ws_price

    def initial_dfs(self, mode: str = 'wicks'):
        return self._renko_df(mode)

    def add_prices(self, ws_timestamp: int, ws_price: float):
        """
        Determine if there are new bricks to add according to the current price relative to the previous renko.

        Must be called at every price change.

        Here, the 'Renko Single Data' is constructed.

        Parameters
        ----------
        ws_timestamp : int
            Timestamp in milliseconds.
        ws_price : float
            Self-explanatory.
        """
        self._ws_timestamp = ws_timestamp
        self._ws_price = ws_price

        self._wick_min_i = ws_price if ws_price < self._wick_min_i else self._wick_min_i
        self._wick_max_i = ws_price if ws_price > self._wick_max_i else self._wick_max_i
        self._volume_i += 1

        last_price = self._rsd["price"][-1]
        current_n_bricks = (ws_price - last_price) / self._brick_size
        current_direction = np.sign(current_n_bricks)
        if current_direction == 0:
            return
        last_direction = self._rsd["direction"][-1]
        is_same_direction = ((current_direction > 0 and last_direction >= 0)
                             or (current_direction < 0 and last_direction <= 0))

        # CURRENT PRICE in same direction of the LAST RENKO
        total_same_bricks = current_n_bricks if is_same_direction else 0
        # >= 2 can be a 'GAP' or 'OPPOSITE DIRECTION'.
        # In both cases we add the current wick/volume to the first brick and 'reset' the value of both, since:
        # If it's a GAP:
        # - The following bricks after first brick will be 'artificial' since the price has 'skipped' that price region.
        # - (the reason of 'totalSameBricks')
        # If it's a OPPOSITE DIRECTION:
        # - Only the first brick will be kept. (the reason of '2' multiply)
        if not is_same_direction and abs(current_n_bricks) >= 2:
            self._add_brink_loop(ws_timestamp, 2, current_direction, current_n_bricks)
            total_same_bricks = current_n_bricks - 2 * current_direction

        # Add all bricks in the same direction
        for not_in_use in range(abs(int(total_same_bricks))):
            self._add_brink_loop(ws_timestamp, 1, current_direction, current_n_bricks)

    def _add_brink_loop(self, ws_timestamp, renko_multiply, current_direction, current_n_bricks):
        last_price = self._rsd["price"][-1]
        renko_price = last_price + (current_direction * renko_multiply * self._brick_size)
        wick = self._wick_min_i if current_n_bricks > 0 else self._wick_max_i

        to_add = [ws_timestamp, renko_price, current_direction, wick, self._volume_i]
        for name, add in zip(list(self._rsd.keys()), to_add):
            self._rsd[name].append(add)

        self._volume_i = 1
        self._wick_min_i = renko_price if current_n_bricks > 0 else self._wick_min_i
        self._wick_max_i = renko_price if current_n_bricks < 0 else self._wick_max_i

    def _renko_df(self, mode: str = "wicks"):
        """
        Transforms 'Renko Single Data' into OHLCV Dataframe.
        """
        if mode not in _MODE_dict:
            raise ValueError(f"Only {_MODE_dict} options are valid.")

        timestamps = self._rsd["timestamp"]
        prices = self._rsd["price"]
        directions = self._rsd["direction"]
        wicks = self._rsd["wick"]
        volumes = self._rsd["volume"]
        brick_size = self._brick_size

        df_dict = {
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        }

        reverse_rule = mode in ["normal", "wicks", "reverse-wicks", "fake-r-wicks"]
        fake_reverse_rule = mode in ["fake-r-nongap", "fake-r-wicks"]
        same_direction_rule = mode in ["wicks", "nongap"]

        prev_direction = 0
        prev_close = 0
        prev_close_up = 0
        prev_close_down = 0
        for price, direction, timestamp, wick, volume in zip(prices, directions, timestamps, wicks, volumes):
            if direction != 0:
                df_dict["timestamp"].append(timestamp)
                df_dict["close"].append(price)
                df_dict["volume"].append(volume)

            # Current Renko (UP)
            if direction == 1.0:
                df_dict["high"].append(price)
                # Previous same direction(UP)
                if prev_direction == 1:
                    df_dict["open"].append(wick if mode == "nongap" else prev_close_up)
                    df_dict["low"].append(wick if same_direction_rule else prev_close_up)
                # Previous reverse direction(DOWN)
                else:
                    if reverse_rule:
                        df_dict["open"].append(prev_close + brick_size)
                    elif mode == "fake-r-nongap":
                        df_dict["open"].append(prev_close_down)
                    else:
                        df_dict["open"].append(wick)

                    if mode == "normal":
                        df_dict["low"].append(prev_close + brick_size)
                    elif fake_reverse_rule:
                        df_dict["low"].append(prev_close_down)
                    else:
                        df_dict["low"].append(wick)
                prev_close_up = price
            # Current Renko (DOWN)
            elif direction == -1.0:
                df_dict["low"].append(price)
                # Previous same direction(DOWN)
                if prev_direction == -1:
                    df_dict["open"].append(wick if mode == "nongap" else prev_close_down)
                    df_dict["high"].append(wick if same_direction_rule else prev_close_down)
                # Previous reverse direction(UP)
                else:
                    if reverse_rule:
                        df_dict["open"].append(prev_close - brick_size)
                    elif mode == "fake-r-nongap":
                        df_dict["open"].append(prev_close_up)
                    else:
                        df_dict["open"].append(wick)

                    if mode == "normal":
                        df_dict["high"].append(prev_close - brick_size)
                    elif fake_reverse_rule:
                        df_dict["high"].append(prev_close_up)
                    else:
                        df_dict["high"].append(wick)
                prev_close_down = price
            # BEGIN OF DICT
            else:
                df_dict["timestamp"].append(np.nan)
                df_dict["low"].append(np.nan)
                df_dict["close"].append(np.nan)
                df_dict["high"].append(np.nan)
                df_dict["open"].append(np.nan)
                df_dict["volume"].append(np.nan)

            prev_direction = direction
            prev_close = price

        df = pd.DataFrame(df_dict)
        # Removing the first 2 lines of DataFrame that are the beginning of respective loops (df_dict and self._rsd)
        df.drop(df.head(2).index, inplace=True)
        # Setting Index
        df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"].values.astype(np.int64), unit="ms"))
        df.index.name = 'datetime'
        df.drop(columns=['timestamp'], inplace=True)

        return df

    def renko_animate(self, mode: str = 'wicks', max_len: int = 500, keep: int = 250):
        """
        Should be called after 'add_prices(ws_timestamp, ws_price)'

        Parameters
        ----------
        mode : str
            The method for building the renko dataframe, described in the Renko.renko_df().
        max_len : int
            Once reached, the 'Single Renko Data' values will be deleted.
        keep : int
            Keep last nº values after deletion.

        Returns
        -------
        x : dataframe
            pandas.Dataframe
        """
        renko_df = self._renko_df(mode)

        ws_timestamp = self._ws_timestamp
        ws_price = self._ws_price

        raw_ws = {
            "timestamp": [ws_timestamp],
            "open": [ws_price],
            "high": [ws_price],
            "low": [ws_price],
            "close": [ws_price],
            "volume": self._volume_i
        }
        length = len(renko_df)
        if length < 1:
            raw_ws["open"][-1] = self.initial_df["close"].iat[-1]
            raw_ws["high"][-1] = self._wick_max_i
            raw_ws["low"][-1] = self._wick_min_i

            df_ws = pd.DataFrame(raw_ws)
            df_ws.index = pd.DatetimeIndex(pd.to_datetime(df_ws["timestamp"].values.astype(np.int64), unit="ms"))
            df_ws.index.name = 'datetime'
            df_ws.drop(columns=['timestamp'], inplace=True)

            return pd.concat([self.initial_df, df_ws])

        # Forming wick
        raw_ws["high"][-1] = self._wick_max_i if mode != 'normal' else ws_price
        raw_ws["low"][-1] = self._wick_min_i if mode != 'normal' else ws_price

        nongap_rule = mode in ['nongap', 'reverse-nongap', 'fake-r-nongap']
        last_renko_close = renko_df["close"].iat[-1]
        last_renko_open = renko_df["open"].iat[-1]
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

        df_ws = pd.DataFrame(raw_ws)
        df_ws.index = pd.DatetimeIndex(pd.to_datetime(df_ws["timestamp"].values.astype(np.int64), unit="ms"))
        df_ws.index.name = 'datetime'
        df_ws.drop(columns=['timestamp'], inplace=True)

        if length >= max_len:
            # Cleaning dictionary, keeping keys and nº last values/bricks
            for value in self._rsd.values():
                del value[:-keep]
            gc.collect()

        return pd.concat([renko_df, df_ws])
