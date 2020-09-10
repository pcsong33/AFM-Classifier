"""Calculates S parameters for any generic dataset of surfaces under a given set
of directories.

Usage:
    extract [--processes=<n>] [-o <output>] [-M <M>] [--Lx <x>] [--Ly <y>] [--raw | --asc ] <input-dir> ...

Help:
    --processes=<n>  Number of processes to use. Defaults to `os.cpu_count() - 1`.
    -o <output> Path to save the file containing S parameters.
    -M <M> For use with spatial parameters. See `dr/parameters/spatial.py` for
    more details.
    --Lx <x>  The lengths along the x-dimension of each surface. Only use
    with `--raw` (i.e. when there is no information about the dimensions of the surface
    included with each file).
    --Ly <y>  The lengths along the x-dimension of each surface. Only use
    with `--raw` (i.e. when there is no information about the dimensions of the surface
    included with each file).
    --raw  Load the files in the input directory using NumPy's `loadtxt(.)`. Default.
    --asc  Load the files in the input directory using `load.load_ascii(.)`.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from docopt import docopt

from dr.load import load_ascii
from dr.extraction import parallelized_extraction_custom, extract_parameters

if __name__ == '__main__':
    # Extract CL arguments
    args = docopt(__doc__)

    # The input directories
    base_paths = [Path(input_dir) for input_dir in args['<input-dir>']]

    # Number of processes to use during parallelized computations
    processes = (max(os.cpu_count() - 1, 1) if args['--processes'] is None
                 else int(args['--processes']))

    # Path to save the results in
    # If ext is '.xlsx', then data is exported as an Excel file.
    # Otherwise, it's a CSV file
    output = 'result.xlsx' if args['-o'] is None else args['-o']
    _, ext = os.path.splitext(output)

    M = 256 if args['-M'] is None else int(args['-M'])

    # `parallelized_extract_custom` takes an `extract` callback that handles the
    # actual extraction for a given file, which we define conditionally here.
    if args['--asc']:
        def extract(fname):
            """Extract callback for .asc files"""
            channel, _, _, dx, dy = load_ascii(fname)
            return extract_parameters(channel, dx, dy, M).rename(fname)
    else:
        def extract(fname):
            """Extract callback for anything else"""
            channel = np.loadtxt(fname)
            X, Y = channel.shape

            if args['--Lx'] and args['--Ly']:
                dx = int(args['--Lx']) / X
                dy = int(args['--Ly']) / Y

                return extract_parameters(channel, dx, dy, M).rename(fname)
            else:
                raise Exception(
                    f"""Cannot infer pixel separation distances for {fname}. 
                            Try passing in --lengths <x> <y> as an option.""")

    dfs = []

    for path in base_paths:
        try:
            _, _, filenames = next(os.walk(path))
        except StopIteration:
            raise Exception(f'Directory {path} does not exist')

        filenames = [str(path.joinpath(filename)) for filename in filenames]
        df = parallelized_extraction_custom(filenames, processes, extract)
        dfs.append(df)

    df = pd.concat(dfs)

    if ext == '.xlsx':
        df.to_excel(output)
    else:
        df.to_csv(output)
