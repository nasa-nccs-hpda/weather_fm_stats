# Repository description

Agentic development of scientific weather model stats. This code is designed to create forecast/reanalysis/climatology datasets from raw netCDF files, then use those three datasets to create global/regional stats files for those datasets. The goal is to create fast, modular code that can support many different models, pressure levels, variables, dates, and so on.

## Workflow

**Note: Workflow runs exclusively on the Discover cluster. The workflow requires a .yaml file to describe what experiment configuration you'd like to run. See `example_yaml_files` for formatting.**

You can either run this code from a Discover login node, or directly from an allocated A100 gpu node. See commands below for examples.

### Option 1: from login node

Format:

```bash
cd path/to/repo/v4
chmod +x sbatch_stats_v4.run
./sbatch_stats_v4.run <yaml_filename>
```

Example command:

```bash
cd $NOBACKUP/weather_fm_stats/v4
chmod +x sbatch_stats_v4.run
./sbatch_stats_v4.run ../example_yaml_files/stats_AIFS_ERA5_MAY_2024.yaml
```

### Option 2: from GPU node

Format

```bash
salloc --job-name=stats --time=8:00:00 --gres=gpu:1 --mem 64G -c 10
cd path/to/repo/v4
chmod +x salloc_run_v4.sh
./salloc_stats_v4.run <yaml filename>
```

Example command:

```bash
salloc --job-name=stats --time=8:00:00 --gres=gpu:1 --mem 64G -c 10
cd $NOBACKUP/weather_fm_stats/v4
chmod +x salloc_run_v4.sh
./salloc_stats_v4.run ../example_yaml_files/stats_AIFS_ERA5_MAY_2024.yaml
```

## Supported models

Supported ML models are GenCast, AIFS, Prithvi. Supported reanalysis/climatology models are GEOSFP, MERRA2, and ERA5.

## Comparison between v1 (original code) and v4

| Experiment | v1 elapsed | v4 elapsed | v4-v1 | v1 files | v4 files |
| :---: | :---: | :---: | :---: | :---: | :---: |
| short | 00:16:05 | 00:17:02 | +00:00:57 | OK | OK |
| long | 00:31:12 | 00:32:06 | +00:00:54 | OK | OK |

The code runs in very similar times, while achieving a much more readable/maintainable code structure. Future work is planned to improve the runtime of the v4 code.
