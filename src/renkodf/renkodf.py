"""
renkodf
=====
Transform Tick Data into OHLCV Renko Dataframe!
"""
import gc

import numpy as np
import pandas as pd
import mplfinance as mpf

_MODE_dict = ['normal', 'wicks', 'nongap', 'reverse-wicks', 'reverse-nongap', 'fake-r-wicks', 'fake-r-nongap']


class Renko:
    def __init__(self, df: pd.DataFrame, brick_size: float, add_columns: list = None):
        """
        Create Renko OHLCV dataframe with existing Ticks data.
        
        Usage
        ------
        >> from renkodf import Renko \n
        >> r = Renko(df_ticks, brick_size) \n
        >> df = r.renkodf() \n
        
        Parameters
        ----------
        df : dataframe
            Only two columns are required:

            * "close": Mandatory.
            * "datetime": If is not present, the index will be used.

        brick_size : float
            Cannot be less than or equal to 0.00...
        add_columns : list
            A list of strings(column names) to be added to the final result, such as spread, quantity, etc.
        """

        if brick_size is None or brick_size <= 0:
            raise ValueError("brick_size cannot be 'None' or '<= 0'")
        if 'datetime' not in df.columns:
            df["datetime"] = df.index
        if 'close' not in df.columns:
            raise ValueError("Column 'close' doesn't exist!")
        if add_columns is not None:
            if not set(add_columns).issubset(df.columns):
                raise ValueError(f"One or more of {add_columns} columns don't exist!")

        self._df = df
        self._brick_size = brick_size
        self._custom_columns = add_columns

        initial_price = (self._df["close"].iat[0] // self._brick_size) * self._brick_size
        # Renko Single Data
        self._rsd = {
            "origin_index": [0],
            "date": [self._df["datetime"].iat[0]],
            "price": [initial_price],
            "direction": [0],
            "wick": [initial_price],
            "volume": [1],
        }
        if add_columns is not None:
            for name in add_columns:
                self._rsd.update({
                    name: [df[name].iat[0]]
                })

        self._wick_min_i = initial_price
        self._wick_max_i = initial_price
        self._volume_i = 1

        for i in range(1, len(self._df["close"])):
            self._add_prices(i)

    def _add_prices(self, i):
        """
        Determine if there are new bricks to add according to the current (loop) price relative to the previous renko.

        Here, the 'Renko Single Data' is constructed.
        """
        if self._df["close"].iat[i] < self._wick_min_i:
            self._wick_min_i = self._df["close"].iat[i]
        if self._df["close"].iat[i] > self._wick_max_i:
            self._wick_max_i = self._df["close"].iat[i]
        self._volume_i += 1

        current_n_bricks = (self._df["close"].iat[i] - self._rsd["price"][-1]) / self._brick_size
        current_direction = np.sign(current_n_bricks)
        if current_direction == 0:
            return

        last_direction = self._rsd["direction"][-1]
        last_price = self._rsd["price"][-1]
        # CURRENT PRICE in same direction of the LAST RENKO
        total_bricks = 0
        if (current_direction > 0 and last_direction >= 0) or (current_direction < 0 and last_direction <= 0):
            total_bricks = current_n_bricks

        # >= 2 can be a 'GAP' or 'OPPOSITE DIRECTION'.
        # In both cases we add the current wick/volume to the first brick and 'reset' the value of both, since:
        # If it is GAP: The following bricks will be 'artificial' since the price has 'skipped' that price region.
        # If it is OPPOSITE DIRECTION: Only the first brick will be kept.
        elif abs(current_n_bricks) >= 2:
            total_bricks = current_n_bricks - 2 * current_direction
            renko_price = last_price + (current_direction * 2 * self._brick_size)
            wick = self._wick_min_i if current_n_bricks > 0 else self._wick_max_i
            volume = self._volume_i

            to_add = [i, self._df["datetime"].iat[i], renko_price, current_direction, wick, volume]
            for name, add in zip(list(self._rsd.keys()), to_add):
                self._rsd[name].append(add)
            if self._custom_columns is not None:
                for name in self._custom_columns:
                    self._rsd[name].append(self._df[name].iat[i])

            self._volume_i = 1
            if current_n_bricks > 0:
                self._wick_min_i = renko_price
            else:
                self._wick_max_i = renko_price

        # Add all bricks in the same direction
        for not_in_use in range(abs(int(total_bricks))):
            last_price = self._rsd["price"][-1]
            renko_price = last_price + (current_direction * 1 * self._brick_size)
            wick = self._wick_min_i if current_n_bricks > 0 else self._wick_max_i
            volume = self._volume_i

            to_add = [i, self._df["datetime"].iat[i], renko_price, current_direction, wick, volume]
            for name, add in zip(list(self._rsd.keys()), to_add):
                self._rsd[name].append(add)
            if self._custom_columns is not None:
                for name in self._custom_columns:
                    self._rsd[name].append(self._df[name].iat[i])

            self._volume_i = 1
            if current_n_bricks > 0:
                self._wick_min_i = renko_price
            else:
                self._wick_max_i = renko_price

        print(f"\r {round(float((i + 1) / len(self._df) * 100), 2)}%", end='')

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
            Modified external dataframe, usually with new columns for plotting indicators, signals, etc.
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

    def renko_df(self, mode: str = "wicks"):
        """
        Transforms 'Renko Single Data' into OHLCV Dataframe.

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

        Returns
        -------
        x : dataframe
            pandas.Dataframe
        """
        if mode not in _MODE_dict:
            raise ValueError(f"Only {_MODE_dict} options are valid.")

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
                df_dict.update({
                    name: []
                })

        prev_close_high = 0
        prev_close_low = 0
        prev_direction = 0
        reverse_high = 0
        reverse_low = 0
        for price, direction, date, wick, volume, index in zip(prices, directions, dates, wicks, volumes, indexes):

            # UP/Bull Renko
            if direction == 1.0:
                df_dict["datetime"].append(date)
                df_dict["high"].append(price)
                df_dict["close"].append(price)
                df_dict["volume"].append(volume)
                if self._custom_columns is not None:
                    for name in self._custom_columns:
                        df_dict[name].append(self._rsd[name][index])

                reverse_high = price - brick_size

                # PREV UP/Bull Renko
                if prev_direction == 1:
                    if mode == "nongap":
                        df_dict["open"].append(wick)
                    else:
                        df_dict["open"].append(prev_close_high)

                    if mode in ["wicks", 'nongap']:
                        df_dict["low"].append(wick)
                    else:
                        df_dict["low"].append(prev_close_high)
                # PREV DOWN/Bear Renko
                else:
                    if mode in ["normal", "wicks", "reverse-wicks", "fake-r-wicks"]:
                        df_dict["open"].append(reverse_low)
                    elif mode == "fake-r-nongap":
                        df_dict["open"].append(prev_close_low)
                    else:
                        df_dict["open"].append(wick)

                    if mode == "normal":
                        df_dict["low"].append(reverse_low)
                    elif mode in ["fake-r-nongap", "fake-r-wicks"]:
                        df_dict["low"].append(prev_close_low)
                    else:
                        df_dict["low"].append(wick)

                prev_close_high = price

            # DOWN/Bear Renko
            elif direction == -1.0:
                df_dict["datetime"].append(date)
                df_dict["low"].append(price)
                df_dict["close"].append(price)
                df_dict["volume"].append(volume)
                if self._custom_columns is not None:
                    for name in self._custom_columns:
                        df_dict[name].append(self._rsd[name][index])

                reverse_low = price + brick_size

                # PREV DOWN/Bear Renko
                if prev_direction == -1:
                    if mode == "nongap":
                        df_dict["open"].append(wick)
                    else:
                        df_dict["open"].append(prev_close_low)

                    if mode in ["wicks", 'nongap']:
                        df_dict["high"].append(wick)
                    else:
                        df_dict["high"].append(prev_close_low)
                # PREV UP/Bull Renko
                else:
                    if mode in ["normal", "wicks", "reverse-wicks", "fake-r-wicks"]:
                        df_dict["open"].append(reverse_high)
                    elif mode == "fake-r-nongap":
                        df_dict["open"].append(prev_close_high)
                    else:
                        df_dict["open"].append(wick)

                    if mode == "normal":
                        df_dict["high"].append(reverse_high)
                    elif mode in ["fake-r-nongap", "fake-r-wicks"]:
                        df_dict["high"].append(prev_close_high)
                    else:
                        df_dict["high"].append(wick)

                prev_close_low = price

            # BEGIN OF DICT
            else:
                df_dict["datetime"].append(np.NaN)
                df_dict["low"].append(np.NaN)
                df_dict["close"].append(np.NaN)
                df_dict["high"].append(np.NaN)
                df_dict["open"].append(np.NaN)
                df_dict["volume"].append(np.NaN)
                if self._custom_columns is not None:
                    for name in self._custom_columns:
                        df_dict[name].append(np.NaN)

            prev_direction = direction

        df = pd.DataFrame(df_dict)
        # Removing the first 2 lines of DataFrame that are the beginning of respective loops (df_dict and self._rsd)
        df.drop(df.head(2).index, inplace=True)
        # Setting Index
        df.index = pd.DatetimeIndex(df["datetime"])
        df.drop(columns=['datetime'], inplace=True)

        return df

    def to_rws(self, use_iloc: int = None):
        """
        Transforms 'Renko Single Data' into a Dataframe,
        which can be used as initial data in the 'RenkoWS' class. \n
        The DatetimeIndex will be converted to Timestamp (from nanoseconds to milliseconds)

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
        rws = pd.DataFrame(self._rsd)
        rws.index = pd.DatetimeIndex(rws["date"]).asi8 // 10 ** 6  # Datetime to Timestamp (ns to ms)
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
            The method for building the external renko df, described in the Renko.renko_df()'.
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
            initial_df.index = pd.DatetimeIndex(pd.to_datetime(initial_df["timestamp"].values.astype(np.int64), unit="ms"))
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

        if ws_price < self._wick_min_i:
            self._wick_min_i = ws_price
        if ws_price > self._wick_max_i:
            self._wick_max_i = ws_price
        self._volume_i += 1

        current_n_bricks = (ws_price - self._rsd["price"][-1]) / self._brick_size
        current_direction = np.sign(current_n_bricks)
        if current_direction == 0:
            return

        last_direction = self._rsd["direction"][-1]
        last_price = self._rsd["price"][-1]
        # CURRENT PRICE in same direction of the LAST RENKO
        total_bricks = 0
        if (current_direction > 0 and last_direction >= 0) or (current_direction < 0 and last_direction <= 0):
            total_bricks = current_n_bricks

        # >= 2 can be a 'GAP' or 'OPPOSITE DIRECTION'.
        # In both cases we add the current wick/volume to the first brick and 'reset' the value of both, since:
        # If it is GAP: The following bricks will be 'artificial' since the price has 'skipped' that price region.
        # If it is OPPOSITE DIRECTION: Only the first brick will be kept.
        elif abs(current_n_bricks) >= 2:
            total_bricks = current_n_bricks - 2 * current_direction
            renko_price = last_price + (current_direction * 2 * self._brick_size)
            wick = self._wick_min_i if current_n_bricks > 0 else self._wick_max_i
            volume = self._volume_i

            to_add = [ws_timestamp, renko_price, current_direction, wick, volume]
            for name, add in zip(list(self._rsd.keys()), to_add):
                self._rsd[name].append(add)

            self._volume_i = 1
            if current_n_bricks > 0:
                self._wick_min_i = renko_price
            else:
                self._wick_max_i = renko_price

        # Add all bricks in the same direction
        for just_for_count in range(abs(int(total_bricks))):
            last_price = self._rsd["price"][-1]
            renko_price = last_price + (current_direction * 1 * self._brick_size)
            wick = self._wick_min_i if current_n_bricks > 0 else self._wick_max_i
            volume = self._volume_i

            to_add = [ws_timestamp, renko_price, current_direction, wick, volume]
            for name, add in zip(list(self._rsd.keys()), to_add):
                self._rsd[name].append(add)

            self._volume_i = 1
            if current_n_bricks > 0:
                self._wick_min_i = renko_price
            else:
                self._wick_max_i = renko_price

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

        prev_close_high = 0
        prev_close_low = 0
        prev_direction = 0
        reverse_high = 0
        reverse_low = 0
        for price, direction, timestamp, wick, volume in zip(prices, directions, timestamps, wicks, volumes):

            # UP/Bull Renko
            if direction == 1.0:
                df_dict["timestamp"].append(timestamp)
                df_dict["high"].append(price)
                df_dict["close"].append(price)
                df_dict["volume"].append(volume)

                reverse_high = price - brick_size

                # PREV UP/Bull Renko
                if prev_direction == 1:
                    if mode == "nongap":
                        df_dict["open"].append(wick)
                    else:
                        df_dict["open"].append(prev_close_high)

                    if mode in ["wicks", 'nongap']:
                        df_dict["low"].append(wick)
                    else:
                        df_dict["low"].append(prev_close_high)
                # PREV DOWN/Bear Renko
                else:
                    if mode in ["normal", "wicks", "reverse-wicks", "fake-r-wicks"]:
                        df_dict["open"].append(reverse_low)
                    elif mode == "fake-r-nongap":
                        df_dict["open"].append(prev_close_low)
                    else:
                        df_dict["open"].append(wick)

                    if mode == "normal":
                        df_dict["low"].append(reverse_low)
                    elif mode in ["fake-r-nongap", "fake-r-wicks"]:
                        df_dict["low"].append(prev_close_low)
                    else:
                        df_dict["low"].append(wick)

                prev_close_high = price

            # DOWN/Bear Renko
            elif direction == -1.0:
                df_dict["timestamp"].append(timestamp)
                df_dict["low"].append(price)
                df_dict["close"].append(price)
                df_dict["volume"].append(volume)

                reverse_low = price + brick_size

                # PREV DOWN/Bear Renko
                if prev_direction == -1:
                    if mode == "nongap":
                        df_dict["open"].append(wick)
                    else:
                        df_dict["open"].append(prev_close_low)

                    if mode in ["wicks", 'nongap']:
                        df_dict["high"].append(wick)
                    else:
                        df_dict["high"].append(prev_close_low)
                # PREV UP/Bull Renko
                else:
                    if mode in ["normal", "wicks", "reverse-wicks", "fake-r-wicks"]:
                        df_dict["open"].append(reverse_high)
                    elif mode == "fake-r-nongap":
                        df_dict["open"].append(prev_close_high)
                    else:
                        df_dict["open"].append(wick)

                    if mode == "normal":
                        df_dict["high"].append(reverse_high)
                    elif mode in ["fake-r-nongap", "fake-r-wicks"]:
                        df_dict["high"].append(prev_close_high)
                    else:
                        df_dict["high"].append(wick)

                prev_close_low = price

            # BEGIN OF DICT
            else:
                df_dict["timestamp"].append(np.NaN)
                df_dict["low"].append(np.NaN)
                df_dict["close"].append(np.NaN)
                df_dict["high"].append(np.NaN)
                df_dict["open"].append(np.NaN)
                df_dict["volume"].append(np.NaN)

            prev_direction = direction

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
            raw_ws["open"][-1] = self.initial_df["close"][-1]
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
        # Last Renko UP/Bull
        if renko_df["close"][-1] > renko_df["open"][-1]:
            if ws_price > renko_df["close"][-1]:
                if mode in ['nongap', 'reverse-nongap', 'fake-r-nongap']:
                    raw_ws["open"][-1] = self._wick_min_i
                else:
                    raw_ws["open"][-1] = renko_df["close"][-1]
                    if mode == "normal":
                        raw_ws["low"][-1] = renko_df["close"][-1]

            else:
                if ws_price < renko_df["open"][-1]:
                    if mode in ['nongap', 'reverse-nongap', 'fake-r-nongap']:
                        raw_ws["open"][-1] = self._wick_max_i
                    else:
                        raw_ws["open"][-1] = renko_df["open"][-1]
                        if mode == "normal":
                            raw_ws["high"][-1] = renko_df["open"][-1]
        # Last Renko DOWN/Bear
        else:
            if ws_price < renko_df["close"][-1]:
                if mode in ['nongap', 'reverse-nongap', 'fake-r-nongap']:
                    raw_ws["open"][-1] = self._wick_max_i
                else:
                    raw_ws["open"][-1] = renko_df["close"][-1]
                    if mode == "normal":
                        raw_ws["high"][-1] = renko_df["close"][-1]
            else:
                if ws_price > renko_df["open"][-1]:
                    if mode in ['nongap', 'reverse-nongap', 'fake-r-nongap']:
                        raw_ws["open"][-1] = self._wick_min_i
                    else:
                        raw_ws["open"][-1] = renko_df["open"][-1]
                        if mode == "normal":
                            raw_ws["low"][-1] = renko_df["open"][-1]

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
