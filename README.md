# Repository description

Agentic development of scientific weather model stats. This code is designed to create forecast/reanalysis/climatology datasets from raw netCDF files, then use those three datasets to create global/regional stats files for those datasets. The goal is to create fast, modular code that can support many different models, pressure levels, variables, dates, and so on.

## Workflow

**Note: Workflow runs exclusively on the Discover cluster. The workflow requires a .yaml file to describe what experiment configuration you'd like to run. See `example_yaml_files` for formatting.**

The normal v4 workflow is a single SLURM job with in-process parallelism for dataset and statistics chunks. Use `sbatch_stats_v4.run` from a login node, or `salloc_stats_v4.run` inside an existing allocation. Lower-level Python flags are intended for debugging, merge recovery, and manual reruns.

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
salloc --job-name=stats --time=8:00:00 --gres=gpu:1 --mem=64G --cpus-per-task=10
cd path/to/repo/v4
chmod +x salloc_stats_v4.run
./salloc_stats_v4.run <yaml filename>
```

Example command:

```bash
salloc --job-name=stats --time=8:00:00 --gres=gpu:1 --mem=64G --cpus-per-task=10 --partition=gpu_a100 --constraint=rome
cd $NOBACKUP/weather_fm_stats/v4
chmod +x salloc_stats_v4.run
./salloc_stats_v4.run ../example_yaml_files/stats_AIFS_ERA5_MAY_2024.yaml
```

## Supported models

Supported ML models are GenCast, AIFS, Prithvi. Supported reanalysis/climatology models are GEOSFP, MERRA2, and ERA5.

## Comparison between original code and v4

Below is an example run comparing long and short experiments for the original "v1" code and the newest "v4" code. There are slight timing differences between each run, due to SLURM scheduling. The newest code runs only slightly slower than the old code, while achieving a much more readable/maintainable code structure and using significantly less compute. Future work is planned to improve the speed of the v4 code.

| Experiment | v1 elapsed | v4 elapsed | v4-v1 | v1 files | v4 files |
| :---: | :---: | :---: | :---: | :---: | :---: |
| short | 00:16:05 | 00:17:02 | +00:00:57 | OK | OK |
| long | 00:31:12 | 00:32:06 | +00:00:54 | OK | OK |
