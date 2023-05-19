from functools import wraps
import sys
import os
import numpy as np
from datetime import datetime
from ast import literal_eval
from time import time


def assert_cur_dir(dir):
    assert os.getcwd().split('/')[-1] == dir


def print_elapsed(msg, start_time, format='min'):
    if format not in ('min', 's'):
        raise ValueError('format must be "min" or "s"')
    elapsed = time() - start_time
    if format == 'min':
        elapsed /= 60
    print('{} in {:.2f} {}'.format(msg, elapsed, format))


def is_tokenized(x):
    """
    :param x: list of texts
    :return boolean
    """
    er_text = "Documents must be a list of strings. The list may or may not be tokenized."
    if not isinstance(x, list):
        raise ValueError(er_text)
    if isinstance(x[0], str):
        return False
    elif isinstance(x[0][0], str):
        return True
    raise ValueError(er_text)


def get_topn_with_indices(sims, n, offset=0):
    """
    :param sims: list or array of floats
    :param n: amount of values to return
    :param offset: amount of top values to ignore
    :return: list of largest floats with their original indices
    >>> get_topn_with_indices([.0, -.2, .1, -.1, .9], 3, offset=1)
    [(2, 0.1), (0, 0.0), (3, -0.1)]
    """
    np_sims = np.array(sims)
    indices = np.argsort(-np_sims)
    top_indices = indices[offset:min(n + offset, len(indices))]
    return list(zip(top_indices, np_sims[top_indices]))


def str_to_list(x):
    """
    >>> str_to_list('[[1], [2, 3]]')
    [[1], [2, 3]]
    """
    return literal_eval(x)


def get_datetime_now():
    return '{0:%Y-%m-%d_%H.%M.%S.%f}'.format(datetime.now())


def flatten(l):
    """
    :param l: list
    :return: flattened list
    >>> flatten([[1, 2], [3]])
    [1, 2, 3]
    """
    if len(l) == 0:
        return l
    return [item for sublist in l for item in sublist]


class LogToFile:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kw):
            old_stdout = sys.stdout
            log_file = open(
                'log/{}_{}.log'.format(self.model_name, get_datetime_now()), 'w')
            sys.stdout = log_file

            res = func(*args, **kw)

            sys.stdout = old_stdout
            log_file.close()
            return res

        return wrapper


class Time:
    def __init__(self, start_msg, end_msg=None, format='min'):
        if format not in ('min', 's'):
            raise ValueError('format must be "min" or "s"')

        self.start_msg = start_msg
        self.end_msg = end_msg
        self.format = format

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kw):
            start = time()
            print(self.start_msg + '...')
            res = func(*args, **kw)
            print_elapsed(self.end_msg or self.start_msg +
                          ' finished', start, self.format)
            return res

        return wrapper


class ForEachETA:
    def __init__(self, iterable, start_msg, end_msg=None):
        self.iterable = iterable
        self.start_msg = start_msg
        self.end_msg = end_msg

    def __call__(self, func):
        @wraps(func)
        def wrapper():
            start_time = time()
            n = len(self.iterable)
            print(self.start_msg)
            A = []
            for i, item in enumerate(self.iterable):
                iter_start_time = time()
                A.append(func(item))
                print('\r{:.4}% done, ETA: {:.2f} min'.format(((i + 1) / n) * 100,
                                                              (n - i - 1) * (time() - iter_start_time) / 60),
                      end='')
            print_elapsed(self.end_msg or self.start_msg +
                          ' finished', start_time)
            return A

        return wrapper
