"""Calculates S parameters for some simulated surfaces. Assumes the surfaces are
stored under <input-dir> and follow the format 'testxxxx.asc'.

Usage:
    simulated <input-dir> <start> <stop> [--processes=<n>] [-o <output>]

Help:
    --processes=<n>  Number of processes to use. Defaults to 4.
    -o <output> Path to save the resulting .csv.
"""
from pathlib import Path

from dr.extraction import parallelized_extraction
from docopt import docopt


# Defaults for Igor's simulated surfaces
DELTA = 10 / 512
M = 256


if __name__ == '__main__':
    args = docopt(__doc__)

    # Extract CL arguments
    base_path = Path(args['<input-dir>'])
    start = int(args['<start>'])
    stop = int(args['<stop>'])
    processes = 4 if args['--processes'] is None else int(
        args['--processes'])
    output = 'result.csv' if args['-o'] is None else args['-o']

    paths = [base_path.joinpath(f'test{i}.asc') for i in range(start, stop)]
    df = parallelized_extraction(paths,
                                 dx=DELTA,
                                 dy=DELTA,
                                 M=M,
                                 processes=processes)
    df.to_csv(output, index=False)
