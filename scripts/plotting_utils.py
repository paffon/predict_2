# plotting_utils.py
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def get_next_color(index: int) -> Tuple[int, str]:
    """
    Get the next color from the stack.

    :param index: Index of the previous color in the stack.
    :return: Index of the next color in the stack and the color.
    """

    colors_stack = ['black', 'red', 'blue', 'green', 'orange',
                    'purple', 'brown', 'pink', 'gray']
    next_index = (index + 1) % len(colors_stack)
    return next_index, colors_stack[index]


def plot_features(data: List[Dict[str, Any]]) -> None:
    """
    Plot multiple data series with titles and axis information.

    :param data: List of dictionaries, each representing a plot with title and data series information.
                 Example:
                 [
                    {
                        'title': 'Moving averages',
                        'data_series': {
                            'main axis': [
                                {'name': 'AAPL Adj. Close', 'data': main_data, 'color': 'black', 'style': '-'},
                                {'name': 'MA 3', 'data': ma_3, 'color': 'green', 'style': '--'},
                                {'name': 'EMA 0.5', 'data': ema_05, 'color': 'orange', 'style': '='}
                            ],
                            'secondary axis': []
                        }
                    },
                    {
                        'title': 'Ratios',
                        'data_series': {
                            'main axis': [
                                {'data': main_data, 'color': 'black', 'style': '-'},
                            ],
                            'secondary axis': [
                                {'name': 'ratio 1',  'data': ratios_1, 'color': 'red', 'style': '--'},
                                {'name': 'ratio 2',  'data': ratios_2, 'color': 'blue', 'style': '='}
                            ]
                        }
                    }
                 ]

    :return: None
    """

    for plot_data in data:
        title = plot_data['title']
        data_series = plot_data['data_series']
        color_idx = 0

        fig, main_axis = plt.subplots()

        for series in data_series['main axis']:
            data = series['data']
            name = series.get('name', data.name)

            if 'color' in series:
                color = series['color']
            else:
                color_idx, color = get_next_color(color_idx)

            style = series.get('style', '-')  # Default to solid line if style is not provided
            main_axis.plot(data, label=name, color=color, linestyle=style)

        main_axis.set_xlabel('X Axis')
        main_axis.set_ylabel('Main Axis Y')
        main_axis.legend(loc='upper left')
        main_axis.set_title(title)

        if data_series['secondary axis']:
            color_idx = 0

            sec_axis = main_axis.twinx()

            for series in data_series['secondary axis']:
                data = series['data']
                name = series.get('name', data.name)

                if 'color' in series:
                    color = series['color']
                else:
                    color_idx, color = get_next_color(color_idx)

                style = series.get('style', '-')  # Default to solid line if style is not provided
                sec_axis.plot(data, label=name, color=color, linestyle=style)

            sec_axis.set_ylabel('Secondary Axis Y')
            sec_axis.legend(loc='upper right')

        plt.show()
