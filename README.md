# Project 1: Data reduction

All data/code relevant to the first phase of the project will (hopefully) be contained here.

## Getting started

1. Verify that you have Python 3.7+ (we make no guarantees for older installations).

```bash
   $ python -V
   Python 3.7.0
```

2. Install all requirements via `pip`.

```bash
    pip install -r requirements.txt
```

## Testing

The command for invoking all tests is ```python -m pytest```.

The reference values used for testing parameters are taken from NanoScope Analysis.
Because that software does not calculate all the parameters featured here, we do 
not include tests for certain parameters.

## Structure of the project

The library for calculating S parameters on a given surface is contained under the directory `dr`, the filetree for which looks like:

- `dr`
    - `parameters`
        - `amplitude.py`
        - `functional.py`
        - `hybrid.py`
        - `spatial.py`
    - `scripts`
        - `organize_zoom_data.py`
        - `simulated.py`
        - `zoom.py`
        - `extract.py`
    - `tests`
        - `test_surface_parameters.py`
    - `extraction.py`
    - `load.py`
    - `utils.py`

`dr/parameters` contains the implementations for each parameter. Every parameter
falls into a class of related parameters, these classes being `amplitude`, 
`functional`, `hybrid`, and `spatial`. For detailed descriptions of each paramter,
consult [this reference](http://www.imagemet.com/WebHelp6/Default.htm#RoughnessParameters/Roughness_Parameters.htm#Radial_Wave_Index_Srwi_).

`dr/utils.py` contains helper functions used in the definitions of each parameter.

`dr/load.py` contains functions for loading channel data from different types of 
files. Of note is the function `load_ascii`, which loads files in the ASC format.
See `data/sample/asc/example0.asc` for an example. For text files that consist entirely
of the channel data (i.e. there are no headers/metadata), `np.loadtxt` will suffice.

`dr/extraction.py` contains functions for extracting the S parameters from a channel
or a set of channels. In particular, `extract_parameters` extracts the parameters
from a single channel, and the various `parallelized_extraction` functions do
this for multiple channels in parallel. 

`dr/test` contains tests. It's not very well fleshed-out right now.

`dr/scripts` contains various scripts, most of which are used for extracting S parameters
from files in a dataset and exporting that data into Excel/CSV formats.

### Extracting parameters from a dataset

Assuming that surfaces for a dataset are all contained under some directory,
`dr/scripts/extract.py` works well for this purpose. It can be invoked as:

```bash
$ python -m dr.scripts.extract <path-to-directory>
```

but you'll probably want to set some options since the script will not know relevant
details like the area of the surface, how many lines/semicircles to draw for calculating
certain spatial parameters, etc. See the docstring in the file or invoke 
`python -m dr.scripts.extract -h` for a list of options.

For the data contained under `data/sample`, we would extract it for each format as
follows:


```bash
# For ASC
$ python -m dr.scripts.extract --asc -o result.xlsx data/sample/asc
```


```bash
# For TXT
$ python -m dr.scripts.extract --raw --Lx 5 --Ly 5 -o result.xlsx data/sample/txt
```

The other scripts contained `dr/scripts` are designed to work with specific datasets
and thus cannot be used generally.

### Notebooks

The `notebooks` directory contains Jupyter notebooks for exploring various phenomena 
related to the parameters. In particular, 

- `notebooks/Surface Simulation Generation.ipynb` generates random cell-like 
surfaces for purposes of testing.

- `notebooks/Areal Independence verification.ipynb` and `notebooks/Areal Independence verification2.ipynb` deal with verifying that our implementations of the S parameters are area-independent 
using different sets of test surfaces. 

- `notebooks/Parameter Correlations.ipynb` calculates S parameters on the set 
of simulated cell surfaces and calculates a cross-correlation matrix on the 
 S parameters.

## Datasets

Due to storage limits, the datasets relevant to the project are not found under 
`data/`. Here the directory is mostly used for storing results and examples.