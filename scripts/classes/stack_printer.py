# A class for this project for visual printing that makes sense
import time


def format_seconds(seconds: float):
    """
    Formats a float as a readable time string.
    If seconds is in 0  incl. to 60 excl., time is formatted as 'ss seconds'
    If seconds in 60 incl. to 3600 excl., time is formatted as 'mm:ss minutes'.
    Anything more is formatted as 'hh:mm:ss hours'.

    :param seconds: float, the time to format
    :return: str, the formatted time string
    """
    if seconds < 60:
        return f'{seconds:.2f} seconds'
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f'{minutes:.0f}:{seconds:.2f} minutes'
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f'{hours:.0f}:{minutes:.0f}:{seconds:.2f} hours'


class StackPrinter:
    """
    A class for printing and timing actions in a nested structure.

    :param sep: Separator used for indentation, default is a single space.
    """

    def __init__(self, sep: str = '   '):
        """
        Initializes a StackPrinter instance.

        :param sep: Separator used for indentation, default is a single space.
        """
        self.sep = sep
        self.stack = []

        self.opener = '{'
        self.closer = '}'

    def print(self, msg):
        """
        Prints a message with proper indentation based on the nested structure.

        :param msg: The message to be printed.
        """
        level_gap = len(self.stack) * self.sep

        msg = msg.replace('\n', '\n' + level_gap)

        print(level_gap + msg)

    def wrap(self, action: str = None):
        """
        Wraps an action, measuring its execution time and printing the result.

        :param action: The action to be wrapped. If None, the last action in the stack is unwrapped.
        """
        if action:
            start_time = time.time()
            new_tuple = (action, start_time)
            self.print(f'{action} {self.opener}')
            self.stack.append(new_tuple)

        else:

            action, start_time = self.stack.pop()

            end_time = time.time()
            duration = end_time - start_time
            readable_duration = format_seconds(duration)

            self.print(f'{self.closer} {action} took {readable_duration}')
