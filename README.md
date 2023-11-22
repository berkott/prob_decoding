# Prob Decoding
A silly description should go here.

## Getting Started
I don't know if everything in the `requirements.txt` is needed right now

## Data
This section describes how to get data

`load_data_from_pids(pids, brain_region, behavior="choice", data_type="all_ks", n_t_bins=30, prior_path=None, t_before=0.5, t_after=1.5, normalize_input=True)`

This function returns data from specific probes.

- `pids`: probe IDs
- `brain_regions`: brain regions to get data from. Options are `'alv', 'ca1', 'cing', 'dg-mo', 'dg-po', 'dg-sg', 'eth', 'fp', 'lp', 'ml', 'or', 'po', 'th', 'visam5', 'visam6a', 'vplpc'`. https://github.com/int-brain-lab/paper-brain-wide-map/blob/main/brainwidemap/meta/region_info.csv
- `behavior` (optional): behavior that we want to get from the animals. Options are `"choice", "prior", "contrast", "reward", "motion_energy", "wheel_velocity", "wheel_speed", "pupil_diameter", "paw_speed"`.
- `data_type` (optional): Options are `'all_ks', 'good_ks', 'thresholded'`. `all_ks` means that we take all sessions from ks = kilosort 2.5 https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1 or maybe https://www.biorxiv.org/content/10.1101/061481v1.full
- `n_t_bins` (optional): number of bins the times are divided up into.
- `prior_path` (optional): 
- `t_before` (optional): number of seconds to include in the sample before the stimulus onset for a given trial
- `t_after` (optional): number of seconds to include in the sample after the stimulus onset for a given trial
- `normalize_input` (optional): If normalized, we get the data as real numbers, if not normalized, we get whole numbers 

Our data choices:
- `pids`: probe IDs
- `brain_regions`: `po`
- `behavior`: `"wheel_speed"`
- `data_type`: `good_ks`
- `n_t_bins`: 40
- `prior_path`: 
- `t_before`: .5
- `t_after`: 1.5
- `normalize_input`: 