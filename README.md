# Prob Decoding
We mainly study decoding (and a bit of encoding) behavior from electrophysiological recordings using Neuropixel arrays from the International Brain Lab (IBL) dataset. We primarily apply a probabilistic machine learning approach.

## Getting Started
If you need to run `src/save_data.ipynb`, you will need to install everything in `requirements.txt`. If you are just loading the data from the `.npy` files, all you need is numpy and whatever libraries you need for data analysis.

## Data
### How to get the data
`load_data_from_pids(pids, brain_region, behavior="choice", data_type="all_ks", n_t_bins=30, prior_path=None, t_before=0.5, t_after=1.5, normalize_input=True)`

This function returns data from specific probes. A description of the important parameters is given below.
- `pids` (str): probe IDs
- `brain_regions` (str): brain regions to get data from. Popular options are `"alv", "ca1", "cing", "dg-mo", "dg-po", "dg-sg", "eth", "fp", "lp", "ml", "or", "po", "th", "visam5", "visam6a", "vplpc"`. See [this link](https://github.com/int-brain-lab/paper-brain-wide-map/blob/main/brainwidemap/meta/region_info.csv) for a better description of all possible regions.
- `behavior` (str, optional): behavior that we want to get from the animals. Options are `"choice", "prior", "contrast", "reward", "motion_energy", "wheel_velocity", "wheel_speed", "pupil_diameter", "paw_speed"`.
- `data_type` (str, optional): Options are `"all_ks", "good_ks", "thresholded"`. `all_ks` means that we take all sessions from kilosort 2.5. See [this paper](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1) or maybe [this paper](https://www.biorxiv.org/content/10.1101/061481v1.full).
- `n_t_bins` (int, optional): number of bins the times are divided up into.
- `t_before` (float, optional): number of seconds to include in the sample before the stimulus onset for a given trial
- `t_after` (float, optional): number of seconds to include in the sample after the stimulus onset for a given trial
- `normalize_input` (bool, optional): If normalized, we get the data as real numbers, if not normalized, we get whole numbers.

### Our data choices:
**For `src/raw_data/small_data`:**

This is data pulled only from pid `dab512bd-a02d-4c1f-8dbc-9155a163efc0`. It contains 367 sessions, each recording from 24 neurons classified as good by Kilosort, over 40 time bins. This data is analyzed in Berkan"s homework 3.

- `pids`: `dab512bd-a02d-4c1f-8dbc-9155a163efc0`
- `brain_regions`: `po`
- `behavior`: `wheel_speed`
- `data_type`: `good_ks`
- `n_t_bins`: 40
- `t_before`: .5
- `t_after`: 1.5
- `normalize_input`: False

**For `src/raw_data/full_data`:**

Make sure to unzip `full_data.zip` into `src/raw_data/full_data`.

This data is pulled from all pids in `src/raw_data/full_data/all_pids.csv` Unfortunately, not all pids were able to be loaded, so only 112 pids are actually used. The 112 that are used are described in `src/raw_data/full_data/names_and_shapes.txt`.

- `pids`: see `src/raw_data/full_data/names_and_shapes.txt`
- `brain_regions`: `po`
- `behavior`: `wheel_speed`
- `data_type`: `all_ks` (note, this is different from `small_data`)
- `n_t_bins`: 40
- `t_before`: .5
- `t_after`: 1.5
- `normalize_input`: False