"""Calculates S parameters for the 'zoom' dataset. Assumes the subdirectories for
each surfaces are stored under <input-dir> and are formatted as '{persid}/{channel}/{...}.asc'

Usage:
    zoom <input-dir> ... [--processes=<n>] [-o <output>] [-M <M>] [--scale=<scale_factor>]

Help:
    --processes=<n>  Number of processes to use. Defaults to 4.
    -o <output> Path to save the resulting .csv.
    -M <M> For use with spatial parameters.
    --scale <scale_factor> Order of 10 by which to divide the data on every surface.
"""
import os
from pathlib import Path

import pandas as pd

from dr.extraction import parallelized_extraction2
from docopt import docopt

RENAMER = {
    'Adhesion Channel': 'a',
    'Height Channel': 'h',
    'Channel 1': 'Ch1',
    'Channel 2': 'Ch2',
    'Channel 3': 'Ch3',
}


if __name__ == '__main__':
    args = docopt(__doc__)

    # Extract CL arguments

    # The input directories
    base_paths = [Path(input_dir) for input_dir in args['<input-dir>']]

    # Number of processes to use during parallelized computations
    processes = 4 if args['--processes'] is None else int(
        args['--processes'])

    # Path to save the results in Excel format
    output = 'result.xlsx' if args['-o'] is None else args['-o']

    M = 256 if args['-M'] is None else int(args['-M'])
    scale_factor = 1 if args['--scale'] is None else int(args['--scale'])

    print(args)

    # Initialize list of DataFrames storing parameters for each path in base_paths
    dfs = []
    for base_path in base_paths:
        _, persid_dirs, _ = next(os.walk(base_path))
        groups = []
        for persid_dir in persid_dirs:
            group = {}
            group['name'] = persid_dir
            _, channel_dirs, _ = next(os.walk(base_path.joinpath(persid_dir)))
            for channel_dir in channel_dirs:
                channel_base_path = base_path.joinpath(persid_dir, channel_dir)
                _, _, fnames = next(os.walk(channel_base_path))

                channel_prefix = RENAMER.get(channel_dir, channel_dir)
                group[channel_prefix] = [
                    str(channel_base_path.joinpath(fname)) for fname in fnames]
            groups.append(group)
        dfs.append(parallelized_extraction2(
            groups, processes, M, f=lambda channel: channel / 10 ** scale_factor))

    df = pd.concat(dfs)
    df.to_excel(output)
