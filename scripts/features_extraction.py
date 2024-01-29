# Module for numerical features extractions
from typing import Tuple

import pandas as pd

import numpy as np


def periodic_ratios(series: pd.Series, gaps: int) -> pd.Series:
    """
    Returns a series of ratios between data point n and n-gaps.

    :param series: Input time series data.
    :param gaps: Number of gaps between data points for ratio calculation.

    :return: Series of ratios between data point n and n-gaps.
    """
    # Check if the length of the series is less than gaps
    if len(series) <= gaps:
        raise ValueError("Gaps exceed the length of the series.")

    # Calculate ratios for each data point n and n-gaps
    ratios_series = series / series.shift(gaps)

    ratios_series.name = f'{series.name}_Ratio_{gaps}'

    return ratios_series


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """
    Returns a series of moving averages of the input series.

    :param series: Input time series data.
    :param window: Size of the moving average window.

    :return: Series of moving averages.
    """
    # Calculate the moving average for each data point
    moving_avg = series.rolling(window).mean()

    moving_avg.name = f'{series.name}_MA_{window}'

    return moving_avg


def exponential_moving_average(series: pd.Series, alpha: float) -> pd.Series:
    """
    Returns a series of exponential moving averages of the input series.

    :param series: Input time series data.
    :param alpha: Smoothing factor.

    :return: Series of exponential moving averages.
    """
    # Calculate the exponential moving average for each data point
    ema = series.ewm(alpha=alpha).mean()

    ema.name = f'{series.name}_EMA_{alpha}'

    return ema


def bollinger_line(data: pd.Series, window: int,
                   std_multiplier: float) -> pd.Series:
    """
    Calculate the Bollinger line for a given window and standard deviation
    multiplier.

    :param data: Input pandas Series containing the data.
    :param window: Size of the moving window for calculating the standard
                   deviation.
    :param std_multiplier: Multiplier for the standard deviation.
                           If positive, it adds the multiplier times the
                           standard deviation.
                           If negative, it subtracts the absolute value of the
                           multiplier times the standard deviation.

    :return: A pandas Series representing the Bollinger line.
    """
    rolling_ma = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    calculated_bollinger_line = rolling_ma + std_multiplier * rolling_std

    calculated_bollinger_line.name = (f'{data.name}_Bollinger_Line_'
                                      f'{window}_{std_multiplier}')

    return calculated_bollinger_line


def ratios_between_seria(data_1: pd.Series, data_2: pd.Series) -> pd.Series:
    """
    Calculate the ratios between two series.

    :param data_1: First series of data
    :param data_2: Second series of data

    :return: A pd Series of the ratios between the two data seria.
    """
    name_1 = data_1.name
    name_2 = data_2.name

    ratios_series = data_1 / data_2

    ratios_series.name = f'Ratio_{name_1}_{name_2}'

    return ratios_series


def days_since_crossing(data_1: pd.Series, data_2: pd.Series) -> pd.Series:
    """
    Calculate the number of days since the last crossover between two
    series.

    :param data_1: First input series.
    :param data_2: Second input series.

    :return: A pandas Series containing the number of days since the last
             crossover.
    """
    df = pd.concat([data_1, data_2], axis=1)

    df['diff_sign'] = np.sign(data_1 - data_2)

    days_since_last_change = [0]
    previous_diff = df.iloc[0]['diff_sign']

    for idx in range(1, len(df)):
        prev_days = days_since_last_change[-1]
        prev_diff = df.iloc[idx - 1]['diff_sign']
        current_diff = df.iloc[idx]['diff_sign']

        if prev_diff == current_diff:
            days_since_last_change.append(prev_days + 1)
        else:
            days_since_last_change.append(0)

    series_from_list = pd.Series(days_since_last_change, index=df.index)
    name_1 = data_1.name
    name_2 = data_2.name
    series_from_list.name = f'Days_Since_Crossover_{name_1}_{name_2}'
    return series_from_list


def ratios_and_days_since_crossing(data_1: pd.Series, data_2: pd.Series
                                   ) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate the ratios between two series and the number of days since
    the last crossover.

    :param data_1: First input series.
    :param data_2: Second input series.

    :return: A tuple containing the ratios between the two series and the
             number of days since the last crossover.
    """
    ratios = ratios_between_seria(data_1, data_2)
    days = days_since_crossing(data_1, data_2)
    return ratios, days


def rsi(data: pd.Series, periods: int) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given pandas Series.

    :param data: The input data series for which RSI is calculated.
    :param periods: The number of periods to use for RSI calculation.

    :return: A pandas Series containing the RSI values.
    """
    # Calculate daily price changes
    delta = data.diff(1)

    # Define gains and losses (positive and negative price changes)
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses over the specified number of periods
    avg_gains = gains.rolling(window=periods, min_periods=1).mean()
    avg_losses = losses.rolling(window=periods, min_periods=1).mean()

    # Calculate relative strength (RS)
    avg_losses_no_zero = (
        avg_losses.replace(to_replace=0, method='ffill')
        .replace(to_replace=0, value=1)
    )
    rs = avg_gains / avg_losses_no_zero

    # Calculate RSI using the formula
    rsi_values = 1 - (1 / (1 + rs))
    rsi_values.name = f'{data.name}_RSI_{periods}'
    return rsi_values


def cyclical_encoding(ratios_series: pd.Series, name: str, index: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    """
    Encode two time series data into a single cyclical encoding.

    :param ratios_series: Series of ratios from 0 to 1.
    :param name: Name of the series.
    :param index: Optional parameter, the index for calculating cyclical encoding.
                  If not provided, the index of ratios_series is used.

    :return: 2 pandas Series containing the cyclical encoding.
    """

    if not isinstance(ratios_series, pd.Series):
        ratios_series = pd.Series(ratios_series, index=index)

    # Use the provided or default index to calculate cyclical encoding
    sin_angle = np.sin(2 * np.pi * ratios_series)
    cos_angle = np.cos(2 * np.pi * ratios_series)

    sin_angle.name = f'{name}_Cyclical_Sine'
    cos_angle.name = f'{name}_Cyclical_Cosine'

    return sin_angle, cos_angle
