# The main class, to handle the entire procedure.
from typing import List

import numpy as np
import pandas as pd

from classes.tickers_data_fetcher import TickersDataFetcher
from classes.stack_printer import StackPrinter
from classes.files_service import FilesService
from classes.my_nn_sk import MyNeuralNetwork

import plotting_utils as pu
import features_extraction as fe


def add_cyclical_encodings(df: pd.DataFrame,
                           requested_cyclical_encodings: List[str]) -> pd.DataFrame:
    if 'weekday' in requested_cyclical_encodings:
        weekly_sin, weekly_cos = fe.cyclical_encoding(
            ratios_series=df.index.weekday / 7,
            name='weekday', index=df.index)
        df['weekday_sin'] = weekly_sin
        df['weekday_cos'] = weekly_cos

    if 'workday' in requested_cyclical_encodings:
        weekly_sin, weekly_cos = fe.cyclical_encoding(
            ratios_series=df.index.weekday / 5,
            name='weekday', index=df.index)
        df['workday_sin'] = weekly_sin
        df['workday_cos'] = weekly_cos

    if 'month' in requested_cyclical_encodings:
        day_of_month = df.index.day
        days_in_month = df.index.days_in_month
        monthly_sin, monthly_cos = fe.cyclical_encoding(
            ratios_series=day_of_month / days_in_month,
            name='monthly', index=df.index)
        df['monthly_sin'] = monthly_sin
        df['monthly_cos'] = monthly_cos

    if 'year' in requested_cyclical_encodings:
        days_in_year = (df.index.is_leap_year * 366 +
                        (~df.index.is_leap_year) * 365)
        yearly_sin, yearly_cos = fe.cyclical_encoding(
            ratios_series=df.index.dayofyear / days_in_year,
            name='yearly', index=df.index)
        df['yearly_sin'] = yearly_sin
        df['yearly_cos'] = yearly_cos

    return df


class ProjectHandler:
    def __init__(self):
        self.sp = StackPrinter()
        self.tf = TickersDataFetcher(stack_printer=self.sp)
        self.fs = FilesService(root='Predict-2', stack_printer=self.sp)

    def get_data(self):
        self.sp.wrap('Project Handler getting data')

        tickers_data_request = [
            {
                'ticker': 'AAPL',  # interested in Apple's ticker
                'start': '2000-07-01',  # start date
                'end': None,  # end date None = Today
                'parameters': ['Adj Close']  # Specified in a list or 'All'
            },
            {
                'ticker': 'LIN',  # interested in Apple's ticker
                'start': '2000-07-01',  # start date
                'end': None,  # end date None = Today
                'parameters': ['Adj Close']  # Specified in a list or 'All'
            },
            {
                'ticker': 'V',  # interested in Apple's ticker
                'start': '2000-07-01',  # start date
                'end': None,  # end date None = Today
                'parameters': ['Adj Close']  # Specified in a list or 'All'
            },
            {
                'ticker': 'META',  # interested in Apple's ticker
                'start': '2000-07-01',  # start date
                'end': None,  # end date None = Today
                'parameters': ['Adj Close']  # Specified in a list or 'All'
            },
            {
                'ticker': 'NEE',  # interested in Apple's ticker
                'start': '2000-07-01',  # start date
                'end': None,  # end date None = Today
                'parameters': ['Adj Close']  # Specified in a list or 'All'
            },
            {
                'ticker': 'TMO',  # interested in Apple's ticker
                'start': '2000-07-01',  # start date
                'end': None,  # end date None = Today
                'parameters': ['Adj Close']  # Specified in a list or 'All'
            },

        ]

        response_data = self.tf.fetch(tickers_data=tickers_data_request)

        for ticker, data in response_data.items():
            self.fs.save_pickle(data=data,
                                folder='data_raw_input',
                                file_name=f'raw_ticker_data_{ticker}')

        self.sp.wrap()

    def extract_features(self):
        all_raw_files = [file
                         for file
                         in self.fs.get_all_files_in_folder('data_raw_input')
                         if file.startswith('raw_ticker_data')]

        for raw_file_name in all_raw_files:
            ticker = raw_file_name.split('_')[-1].split('.')[0]

            main_data_series = self.fs.load_pickle(folder='data_raw_input',
                                                   file_name=raw_file_name)

            main_data_series = main_data_series[main_data_series.columns[0]]

            main_data_series.name = ticker + '_' + main_data_series.name

            self.sp.wrap(f'Extracting features for {ticker}')
            self.extract_features_for_ticker_and_save(main_data_series, ticker)
            self.sp.wrap()

    def extract_features_for_ticker_and_save(self, main_data_series, ticker):

        # Simple periodic rations
        periodic_ratios_dict = {
            f'periodic_ratios_{i}': fe.periodic_ratios(main_data_series,
                                                       gaps=i)
            for i in range(5)}

        # Simple moving averages
        ma_dict = {}
        for window in [10, 12, 26, 30]:
            ma = fe.moving_average(main_data_series, window=window)
            ratios, days = fe.ratios_and_days_since_crossing(main_data_series, ma)
            serial = f'{window}'
            ma_dict[f'ma_{serial}_ratios'] = ratios
            ma_dict[f'ma_{serial}_days'] = days
        print('cp3')
        # Exponential moving averages
        ema_dict = {}
        for alpha in range(1, 10):
            alpha_float = float(alpha) / 10
            ema = fe.exponential_moving_average(main_data_series, alpha=alpha_float)
            ratios, days = fe.ratios_and_days_since_crossing(main_data_series, ema)
            serial = f'{alpha_float}'
            ema_dict[f'ema_{serial}_ratios'] = ratios
            ema_dict[f'ema_{serial}_days'] = days

        # RSI index
        rsi_dict = {f'rsi_{i}': fe.rsi(main_data_series, periods=i)
                    for i in [10, 14, 20]}

        # Bollinger bands
        bollinger_dict = {}
        for window in [10, 12, 26, 30]:
            bollinger_dict[window] = {}
            for std in [1, 2, 3]:
                bollinger = fe.bollinger_line(main_data_series,
                                              window=window,
                                              std_multiplier=std)
                ratios, days = fe.ratios_and_days_since_crossing(
                    main_data_series,
                    bollinger)
                serial = f'{window}_{std}'
                bollinger_dict[f'bollinger_{serial}_ratios'] = ratios
                bollinger_dict[f'bollinger_{serial}_days'] = days

        extracted_features = {}
        extracted_features.update(periodic_ratios_dict)
        extracted_features.update(rsi_dict)
        extracted_features.update(ma_dict)
        extracted_features.update(ema_dict)

        # Uncomment this to plot the extracted features

        plots_data = [
            {
                'title': 'Moving averages',
                'data_series': {
                    'main axis': [
                        # {'data': main_data_series},
                        {'data': ema_dict['ema_0.2_ratios']},
                        # {'data': ema_095},
                    ],
                    'secondary axis': [
                        # {'data': days_since_data_and_ema_crossed, 'color': '#999999', 'style': ':'}
                        # {'data': weekly_sin, 'color': 'red', 'style': '--'},
                        # {'data': weekly_cos, 'color': 'red', 'style': '--'},
                        # {'data': monthly_sin, 'color': 'blue', 'style': '--'},
                        # {'data': monthly_cos, 'color': 'blue', 'style': '--'},
                        # {'data': yearly_sin, 'color': 'orange', 'style': '--'},
                        # {'data': yearly_cos, 'color': 'orange', 'style': '--'},
                    ]
                }
            },
            {
                'title': 'Ratios',
                'data_series': {
                    'main axis': [
                        {'name': 'AAPL Adj. Close', 'data': main_data_series},
                    ],
                    'secondary axis': [
                        # {'name': 'ratio 1',  'data': ratios_1, 'color': 'red'},
                        # {'name': 'ratio 2',  'data': ratios_2, 'color': 'blue'}
                    ]
                }
            }
        ]

        # pu.plot_features(plots_data)

        saving_instructions = [
            {
                'shifts': [0],
                'prefix': 'y',
                'data': [
                    np.sign(fe.periodic_ratios(main_data_series, gaps=1) - 1)
                ]
            },
            {
                'shifts': range(10),
                'prefix': 'x',
                'data': extracted_features.values()
            },
        ]

        for instruction in saving_instructions:
            prefix = instruction.get('prefix', '')
            shifts = instruction.get('shifts', [0])
            data_seria_to_be_saved = instruction['data']

            for i, data in enumerate(data_seria_to_be_saved):
                for shift in shifts:
                    shifted_data = data.shift(shift)
                    enumeration = str(i).rjust(len(str(len(data_seria_to_be_saved))), '0')
                    shifted_name = f'{prefix}_{ticker}_{enumeration}_{data.name}_shifted_{shift}'

                    shifted_data.name = shifted_name

                    self.fs.save_pickle(data=shifted_data,
                                        folder='data_raw_input',
                                        file_name=shifted_name)

    def train(self):
        clean_x, clean_y = self.get_clean_data()

        my_nn = MyNeuralNetwork(input_len=len(clean_x.columns))

        my_nn.train(clean_x, clean_y)

        self.fs.save_pickle(data=my_nn.model,
                            folder='data_training_outcome',
                            file_name='model')

        prediction = my_nn.predict(clean_x)

        # Create a DataFrame with columns 'Prediction' and 'Clean_y'
        df_real_vs_prediction = pd.DataFrame({
            'Real': clean_y,
            'Prediction': prediction.flatten(),
        })

        self.fs.save_pickle(data=df_real_vs_prediction,
                            folder='data_training_outcome',
                            file_name='real_vs_prediction')

    def get_clean_data(self):
        all_files = self.fs.get_all_files_in_folder('data_raw_input')

        x_names = [file for file in all_files if file.startswith('x_')]
        y_names = [file for file in all_files if file.startswith('y_')]

        # raw = self.fs.load_pickle('data_raw_input', 'AAPL_ticker_data')
        xs = [self.fs.load_pickle('data_raw_input', name) for name in x_names]
        y = self.fs.load_pickle('data_raw_input', y_names[0])

        # Preprocess your data, handle missing values, etc.
        # Extract column names from the names of the Series
        column_names = [series.name for series in xs]
        # Create a DataFrame from the list of pandas Series
        x_df = pd.DataFrame({name: series for name, series in zip(column_names, xs)})

        x_df = add_cyclical_encodings(x_df, ['workday', 'month', 'year'])

        x_and_y = x_df.assign(y=y.shift(-1))
        x_and_y = x_and_y.dropna()

        clean_x = x_and_y.drop(columns=['y'])
        clean_y = x_and_y['y']

        return clean_x, clean_y

    def report_training_outcome(self):
        df_real_vs_prediction = self.fs.load_pickle(
            folder='data_training_outcome',
            file_name='real_vs_prediction')

        df_real_vs_prediction['equal'] = df_real_vs_prediction['Real'] == df_real_vs_prediction['Prediction']

        correct_predictions = df_real_vs_prediction['equal'].sum()
        total_predictions = len(df_real_vs_prediction)

        success_rate = correct_predictions / total_predictions

        print(f'Accuracy: {100 * round(success_rate, 2)}%')
        pass
