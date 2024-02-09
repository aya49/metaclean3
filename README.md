# MetaClean3.0

MetaClean3.0 cleans/removes low-quality events in [FCS (flow cytometry standard)](https://isac-net.org/page/Data-Standards) data detected during irregular flow.

This method is developed for and funded by [Metafora biosystem](https://www.metafora-biosystems.com/)'s [MetaFlow](https://www.metafora-biosystems.com/metaflow/) platform.

## Installation

**Pre-requisites**:

- [Python 3](https://www.python.org/)
- [PyTorch](https://pytorch.org/) and [cuML](https://docs.rapids.ai/api/cuml/stable/execution_device_interoperability/#Installation) (Optional, for processing on GPU)

To install Python 3 on Linux:

```console
metaclean3_public$ sudo apt install python3
metaclean3_public$ sudo apt install python3-pip
```

**Install this package**:

Install this package via PyPI:

```console
metaclean3_public$ pip install metaclean3
```

OR install this package locally:

1. Download this repository.
2. Build and install the package; on Linux, go to the repository directory:

```console
metaclean3_public$ python -m pip install pip-tools
metaclean3_public$ pip-compile pyproject.toml # generate requirements.txt
metaclean3_public$ pip-sync # install dependencies from generate requirements.txt
metaclean3_public$ pip install .
```

## Usage

See API [here](https://metaclean3-metafora-biosystems-public-2521aa7e9f24ff31e1e5a27a1.gitlab.io/autoapi/metaclean3/index.html).

MetaClean3.0 contains three classes, to of which you will directly interact with:

![MetaClean3.0 classes](package_classes.png)

Once you have installed `metaclean3`, in Python:

1. Load your .fcs file using the [fcsparser](https://pypi.org/project/fcsparser/) package. (Optional) We recommend that you compensate and transform your file before applying MetaClean3.0 for optimal results.

```python
import fcsparser

fcs_file_local = '/path/to/fcs_file.fcs'
meta, data = fcsparser.parse(fcs_file_local, reformat_meta=False)
```

2. Instantiate an FCSfile class `f` that will help format your .fcs file for MetaClean3.0. We strongly recommend users to compensate (and optionally transform) the .fcs file before applying MetaClean3.0.

```python
from metaclean3 import FCSfile
from metaclean3.utils import (
    get_timestep,
    get_spillover_raw,
    apply_compensation_matrix
)

# compensate .fcs file
sm = get_spillover_raw(meta=meta, dat_columns=data.columns)
if not (sm is None):
    data[sm.columns] = apply_compensation_matrix(data[sm.columns], sm)

# instantiate FCSfile
f = FCSfile(data=data, time_step=get_timestep(meta))
```

3. Run MetaClean3.0 on your file `f`.

```python
from metaclean3 import MetaCleanFCS

d = MetaCleanFCS().apply(fcs=f)
```

If you wish, MetaClean3.0 can save cleaning process **plots** to a directory of your choice:

```python
png_dir = '/path/to/save/png/plots/in'
d = MetaCleanFCS(png_dir=png_dir).apply(fcs=f)
```

The output `pandas.DataFrame`, `d` contains additional columns:

- `clean_keep`: The boolean column indicating which events to keep (`True`) or remove (`False`). This is the final result you are looking for!
- `bin`: The bin label for each event.
- `outlier_keep`: The boolean column indicating which events were deemed as outliers (`False`).
- `val_*`: The feature values extracted based on the chosen fluorescent measurement.

### General-purpose cleaning

If you already have your way of extracting features and binning your .fcs data and you just want to execute the cleaning portion of MetaClean3.0 on a `pandas.DataFrame`, `binned_data`, you can do so directly with `MetaClean`:

```python
from metaclean3 import MetaClean

d = MetaClean().clean(data=binned_data) # all columns will be used.
```

## Fine-tuning MetaClean3.0

For most cases, we recommend using default settings. However, if there are cases when you want to fine-tune results, some common arguments you can change are listed below. See attributes in the [`MetaCleanFCS`](https://metaclean3-metafora-biosystems-public-2521aa7e9f24ff31e1e5a27a1.gitlab.io/autoapi/metaclean3/index.html#metaclean3.MetaCleanFCS) and [`MetaClean`](https://metaclean3-metafora-biosystems-public-2521aa7e9f24ff31e1e5a27a1.gitlab.io/autoapi/metaclean3/index.html#metaclean3.MetaClean) for more details.

### `FCSfile` arguments control how events are binned

See `FCSfile` API [here](https://metaclean3-metafora-biosystems-public-2521aa7e9f24ff31e1e5a27a1.gitlab.io/autoapi/metaclean3/index.html#metaclean3.FCSfile).

`min_bin_size` and `max_bin_size` (default: `2000` and `10000`): The minimum and maximum number of bins allowed. If you have a large file with more than ten million events and you want MetaClean3.0 results to be more precise, you can increase `max_bin_size`. Adjust with moderation.

```python
f = FCSfile(data=data, time_step=get_timestep(meta), min_bin_size=2000, max_bin_size=10000)
```

### `MetaCleanFCS` arguments control feature selection and generation

See `MetaCleanFCS` API [here](https://metaclean3-metafora-biosystems-public-2521aa7e9f24ff31e1e5a27a1.gitlab.io/autoapi/metaclean3/index.html#metaclean3.MetaCleanFCS).

`fluo_chans_no` (default: `4`): The number of fluorescent measurements to consider.

```python
d = MetaCleanFCS(fluo_chans_no=4).apply(fcs=f)
```

`fluo_chans_clean` (default: `None`): If you already know what fluorescent measurements you want MetaClean3.0 to refer to when determining what events to keep, you can list them here as a string list. To see what fluorescent measurements are in your file, refer to your `FCSfile` instance, `f.fluo_chans`.

```python
d = MetaCleanFCS(fluo_chans_clean=['FL1', 'FL2']).apply(fcs=f)
```

`candidate_chans_type` (default: `fluo`): The type of channels to use for cleaning i.e. `fluo` (fluorescent), `phys` (physical morphology), or `all`.

```python
d = MetaCleanFCS(candidate_chans_type=fluo).apply(fcs=f)
```

`rm_outliers` (default: 'all'): MetaClean3.0 detects and removes outliers so they do not skew MetaClean3.0's judgement when removing low-quality events. If, in the final results, you do not want to remove all of these outliers, you can specify to keep them by setting `rm_outliers` to `'all'`. If you want to keep some of the less outlying events, set this parameter to 'some' and if you want to remove all outliers, set this parameter to 'none'.

```python
d = MetaCleanFCS(rm_outliers='all').apply(fcs=f)
```

`n_cores` (default: `-1`): The number of cores to use while calculating the density feature. Set to -1 to use all cores.

```python
d = MetaCleanFCS(n_cores=-1).apply(fcs=f)
```

### `MetaClean` arguments control degree of leniency

See `MetaClean` API [here](https://metaclean3-metafora-biosystems-public-2521aa7e9f24ff31e1e5a27a1.gitlab.io/autoapi/metaclean3/index.html#metaclean3.MetaClean).

`min_ref_percent` (default: `0.4` 40\%; range: `[0, 1]`): The minimum percentage of bins they want MetaClean3.0 to keep.

```python
d = MetaCleanFCS(min_ref_percent=0.4).apply(fcs=f)
```

`min_ref_percent_to_keep` (default: `0.4` 40\%; range: `[0, 1]`): If non-reference segments contain at least `min_ref_percent_to_keep` bins, we keep it even if it differs from the longest reference segment.

```python
d = MetaCleanFCS(min_ref_percent_to_keep=0.4).apply(fcs=f)
```

`p_thres` (default: `0.05`; range: `[0, 1]`): Adjacent event segments (along time) that do not have significantly different values (i.e. p-value > `p_thres`) are merged. If `p_thres` is small, MetaClean3.0 will remove fewer events (i.e. be more lenient).

```python
d = MetaCleanFCS(p_thres=0.05).apply(fcs=f)
```

`percent_shifts` (default: `[0.15, 0.2, 0.25, 0.3, 0.35, 0.4]`; range: `[0, 1]`): Adjacent event segments (along time) that have quantile values within a specified range of each other are merged. `percent_diff=0.05` indicates that the quantiles to be tested are 5s\% and 95\%. `percent_shifts=[0.15, 0.2]` indicates that MetaClean3.0 will test if the segments have quantile values within a 15\% and 20% range of each other, whiever yeilds results that achieve a higher silhouette score. Small `percent_shift` values mean that MetaClean3.0 will remove fewer events (i.e. be more lenient).

```python
d = MetaCleanFCS(percent_shift=[0.15]).apply(fcs=f)
```

## API/Documentation

See API [here](https://metaclean3-metafora-biosystems-public-2521aa7e9f24ff31e1e5a27a1.gitlab.io/autoapi/metaclean3/index.html).

This repository uses [Sphinx](https://www.sphinx-doc.org/) to generate documentation contained in the `docs`. To generate documentation locally on Linux:

1. install Sphinx for Python:

```console
metaclean3_public$ apt-get install python3-sphinx
metaclean3_public$ pip3 install -r docs/source/requirements.txt
metaclean3_public$ # sphinx-quickstart docs
```

2. Generate the HTML documentation:

```
metaclean3_public$ sphinx-build -M html docs/source/ docs/build/
```

Now you can open the front page in your web browser: `docs/build/html/index.html`.

## Development

**Unit testing**: put test .fcs files in `tests/data` before starting.

```console
metaclean3_public$ python3 -m unittest discover tests
```

**Build and compile package**:

```console
metaclean3_public$ python -m pip install pip-tools
metaclean3_public$ pip-compile --extra dev pyproject.toml # generate requirements.txt
metaclean3_public$ pip install -r requirements.txt
metaclean3_public$ python3 -m build # builds package into a zip
metaclean3_public$ twine check dist/* # check if package description renders on PyPI
```

**Upload to TestPyPI**:

```console
metaclean3_public$ twine upload -r testpypi dist/* # upload
metaclean3_public$ python -m pip install -i https://test.pypi.org/simple metaclean3 # test install
```

**Upload package to PyPI**:

```console
metaclean3_public$ twine upload dist/*
metaclean3_public$ pip install metaclean3 # test install
```

## Support

See the 'Issues' section of this repository.

## Citation

If this package helped you in any way please cite MetaClean3.0's paper:

(TBP)
