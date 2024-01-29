from typing import List

import pandas as pd
import yfinance as yf
from classes.stack_printer import StackPrinter

from datetime import datetime, timedelta


class TickersDataFetcher:
    """
    An interface between the current project and yfinance's API
    """

    def __init__(self, stack_printer: StackPrinter = StackPrinter()):
        self.sp = stack_printer

    def fetch(self, tickers_data: List[dict]) -> dict[str, pd.DataFrame]:
        """
        Fetches the data for a single ticker
        """
        result = {}
        for ticker_data in tickers_data:
            result_data = self.fetch_singular_ticker(ticker_data)
            result[ticker_data['ticker']] = result_data

        return result

    def fetch_singular_ticker(self, request_data: dict) -> pd.DataFrame:
        """
        Gets specified parameters for a given ticker, date range.

        :param request_data: a dict with specifications for the requested data.
                             It should have the keys:
                             - 'ticker' (str),
                             - 'start' (str, format: 'YYYY-MM-DD'),
                             - 'end' (str, format: 'YYYY-MM-DD'),
                             - 'parameters' (List[str])
        :return: A dataframe with the requested data from yfinance.
        """

        # Log the action
        self.sp.wrap(f'Getting data for {request_data["ticker"]}')

        # Extract parameters from request_data
        ticker_str = request_data['ticker']

        # Process start date
        if request_data['start'] is None:
            # Default to one year ago if not specified
            date_from = datetime.now() - timedelta(days=365)
        else:
            date_from = datetime.strptime(request_data['start'], '%Y-%m-%d')

        # Process end date
        if request_data['end'] is None:
            # Default to today if not specified
            date_to = datetime.now()
        else:
            date_to = datetime.strptime(request_data['end'], '%Y-%m-%d')

        # a list of strings, could be ['Close', 'High', 'Volume']
        parameters = request_data['parameters']

        # Fetch data from yfinance
        result_df = yf.download(ticker_str, start=date_from, end=date_to)

        if isinstance(parameters, list):
            result_df = result_df[parameters].copy()

        # Log completion
        self.sp.wrap()

        return result_df
