"""This module contains functions for extracting a surface as a NumPy array from
various file formats.

Given a text file consisting purely of the data describing a surface (i.e. no headers
or metadata), `np.loadtxt` works well.
"""

import os
import re
import linecache
from io import StringIO

import numpy as np
import pandas as pd


def pad_to_quadratic(channel: np.array) -> np.array:
    """Pads a 2D array with as many zeroes in either dimension s.t. the
    resulting array is quadratic in its shape (i.e. square).

    Args:
        channel (np.array): An MxN array.

    Returns:
        np.array: An MxM or an NxN array, whichever is larger.
    """
    M, N = channel.shape
    max_dim = max(M, N)
    return np.pad(channel, ((0, max_dim - M), (0, max_dim - N)))


def load_table(file: str) -> pd.DataFrame:
    # Open .txt file; remove all extra whitespace and paranthesized expressions
    with open(file, 'r+') as fp:
        text = re.sub(' {2,}', ' ', fp.read())
        text = re.sub('\([^ (]*\)', '', text)

    # Load table into Dataframe, excluding a junk column
    pd_table = pd.read_table(StringIO(text), sep=' ')
    pd_table = pd_table[pd_table.columns[:-1]]

    # Verify that no values are NaN
    if np.sum(pd_table.isna().values) != 0:
        raise Warning('load_table: null values in table')
    return pd_table


def load_ascii(file: str, sentinel_string='# Start of Data:') -> (np.array, int, int, float, float):
    """Loads a channel from an ASC file

    Args:
        file (str): Path to the ASC file containing data on a surfradce.
        sentinel_string (str, optional): String indicating the start of data. Everything
        after the occurrence of this string is assumed to be data for the channel. 
        Defaults to '# Start of Data:'.

    Returns:
        np.array: The channel
        int: The number of pixels along the x-dimension
        int: The number of pixels along the y-dimension
        float: The pixel separation distance along the x-dimension
        float: The pixel separation distance along the y-dimension
    """
    with open(file, 'r') as fp:
        contents = fp.read()
        idx = contents.find(sentinel_string)
        data = contents[idx + len(sentinel_string):]

    channel = np.loadtxt(StringIO(data))
    M, N = channel.shape

    # The surface should be square, but if not we pad it with zeros so that it is
    # This affects the calculations for dx and dy, but practically not by much
    # since the instances where M != N are typically just off by 1 (e.g. 256 vs 255)
    if M != N:
        channel = pad_to_quadratic(channel)
        M, N = channel.shape

    # x-length
    Lx = float(re.sub("[^0-9.]", '', linecache.getline(file, 7)))
    # y-length
    Ly = float(re.sub("[^0-9.]", '', linecache.getline(file, 8)))

    dx = (Lx / 1_000) / M
    dy = (Ly / 1_000) / N

    return channel, M, N, dx, dy


if __name__ == '__main__':
    # Example for an ASC file
    data = np.loadtxt('simulated/test0.asc')
