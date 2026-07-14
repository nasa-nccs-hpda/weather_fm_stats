# Repository description

Agentic development of scientific weather model stats. This code is designed to create forecast/reanalysis/climatology datasets from raw netCDF files, then use those three datasets to create global/regional stats files for those datasets. The goal is to create fast, modular code that can support many different models, pressure levels, variables, dates, and so on.

## Workflow

**Note: Workflow runs exclusively on the Discover cluster. The workflow requires a .yaml file to describe what experiment configuration you'd like to run. See `example_yaml_files` for formatting.**

The normal workflow is a single SLURM job with in-process parallelism for dataset and statistics chunks. Use `sbatch_stats.run` from a login node, or `salloc_stats.run` inside an existing allocation. Lower-level Python flags are intended for debugging, merge recovery, and manual reruns.

Pipeline outputs are written to `outputs/` at the repository root.

### Running the workflow

**Format:**

```bash
cd path/to/repo
chmod +x sbatch_stats.run
./sbatch_stats.run <yaml_filename>
```

**Example command:**

```bash
cd $NOBACKUP/weather_fm_stats
chmod +x sbatch_stats.run
./sbatch_stats.run example_yaml_files/short_exp/AIFS_ERA5_ERA5_MAY_2024.yaml
```

## Supported models

Supported ML models are GenCast, AIFS, Prithvi. Supported reanalysis/climatology models are GEOSFP, MERRA2, and ERA5. Note: different models will have different file globbing/date organization patterns. To have these loaded correctly, look at the example .yaml files for help.

Variable availability differs by model and by the chosen analysis/climatology pair. The table below describes the currently validated example YAMLs. The short examples use only `T` and `T2m`; the long examples use the broadest supported variable set for that model combination.

| Example model combination | Long-example supported variables | Long-example unsupported or omitted variables | Notes |
| :--- | :--- | :--- | :--- |
| `AIFS` / `ERA5` / `ERA5` | `Q`, `T`, `U`, `V`, `Z`, `T2m`, `U10m`, `V10m`, `D2m`, `P`, `PS` | None currently omitted from the shared long variable set | Baseline example for the full variable set. |
| `GenCast` / `ERA5` / `ERA5` | `Q`, `T`, `U`, `V`, `Z`, `T2m`, `U10m`, `V10m`, `P` | `D2m`, `PS` | GenCast files provide `H`, `QV`, `T`, `U`, `V`, `SLP`, `T2M`, `U10M`, and `V10M`. The YAML maps `Z` through `H`, maps `Q` through `QV`, and maps `P` through `SLP`. |
| `Prithvi` / `MERRA2` / `MERRA2` | `Q`, `T`, `U`, `V`, `Z`, `T2m`, `U10m`, `V10m`, `PS` | `P`, `D2m` | Current MERRA2 analysis files do not expose `P`/`SLP`; current MERRA2 climatology slice files do not expose `D2m`. |

## Modifying YAML Files

Start from the closest file in `example_yaml_files/short_exp` or `example_yaml_files/long_exp`, then edit only the fields needed for the experiment.

To change the date range, update `start_date`, `end_date`, `fcst_length`, `fcst_interval`, and `fcst_spacing`. Use `exclude_dates` when specific initialization dates should be skipped.

To change models, update `fcst_model`, `ana_model`, `clim_model`, and the matching input directories/templates. Model-specific file layouts vary, so copy the path/template block from an existing example for that model when possible. Also update `expver` to match the forecast model name and set `verify` to the intended verification label.

To change variables, edit `3d_vars_default`, `2d_vars_default`, and `2d_vars_slices`. Keep variables in the collection where the model actually provides them. For example, some models provide `P` in the default 2D collection while others expose surface variables through `slices`. If the same comparison variable is stored under a different source name, keep the requested output variable name and add or adjust aliases.

To add aliases, edit the `*_alias` lists near the bottom of the YAML. Aliases are case-insensitive names that may appear in source NetCDF files. The example YAMLs share a broad alias set so the same requested variable can be found across models with different naming conventions.

Other common configuration options include `regions`, `stats_types`, `dir_loc` for searching existing processed datasets, `pipeline_cpus`, `pipeline_mem`, worker counts, chunk sizes, and `pipeline_log_level` (normal, verbose, or debug).

**Some model/variable/date combinations may fail because files are missing, variables are unavailable, pressure levels do not match, or a requested variable cannot be calculated from available dependencies. The pipeline logs include validation errors that identify what went wrong, such as missing files, missing variables, missing pressure levels, or failed calculated-variable dependencies.**

## Comparison between original code and current pipeline

Below is an example run comparing long and short experiments for the archived original `v1` code and the current single-job pipeline. There are slight timing differences between each run, due to SLURM scheduling. This experiment was run on 2026-07-09. In this run, the current pipeline was faster for both example workflows while keeping the cleaner single-job structure.

| Experiment | v1 elapsed | current elapsed | current-v1
| :---: | :---: | :---: | :---:
| short | 00:10:40 | 00:04:00 | -00:06:40
| long | 00:27:11 | 00:19:01 | -00:08:10
