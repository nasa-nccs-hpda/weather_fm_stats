# Repository description

Working repo for creating weather model stats. This code is designed to create forecast/reanalysis/climatology datasets from raw netCDF files, then use those three datasets to create global/regional stats files for those datasets. The goal is to create fast, modular code that can support many different models, pressure levels, variables, dates, and so on.

## Main changes to repo from original source code

##### Change 1: parallelization structure

SLURM array job parallelization was changed to use a single SLURM job, and parallelize within the Python code.

  - Dataset parallelization is implemented differently for each dataset type (parallelization code is found under `model/dataset_parallel_executor.py`). fcst datasets use chunked processing by init_date;  data also uses chunking by valid_date; clim chunking was found to be unstable, and was left unparallelized.
  - Stats parallel execution uses chunking by init_date, similar to fcst datasets (code is found under `statistics_parallel_executor.py`). Regional and global stats do NOT run in parallel though, this was found to be unstable as well. Regional stats run first, then global.
  - Chunking behavior (used by fcst/ana datasets and both stats types) is largely controlled by the parallel executor files (`model/dataset_parallel_executor.py`, `model/statistics_parallel_executor.py`). These are supported by the `model/chunk_plan.py` file (contains formulaic chunking code), as well as the `model/worker_controls.py` (determines Python parallel worker counts based on YAML file and compute given by SLURM).

##### Change 2: code structure

Behaviors were moved from being in a single file to being in several focused files. The common "model/view/controller" code organization setup was used to inspire the structure of this code.

  - Model code controls the actual scientific behavior of the code.
  - View code is usually intended for a user interface or similar interactable code, but this is left virtually empty since we are using the command-line.
  - Controller code serves as the high-level "orchestration" of the code; it organizes how the code flows from start to finish, and leaves the details to the model code.

##### Change 3: code modularity/object-oriented code

Code was made to be more modular, using object-oriented programming. This means that for certain tasks, we define a class (or "object") to perform that task and that task only. Previously, we had classes like BatchDatasetProcessor and StatisticsProcessor that performed many different tasks at once. This has been streamlined, so that when people wish to edit a single behavior of the code (such as how we regrid variables, or how we parallelize dataset creation) they can just edit a single class that performs that single behavior.

  - **Note:** this point is mostly true for this version of the code, but not every single part of the code is entirely modular in the most recent code version. This was mainly done in the interest of time, but to write the most modular code more behaviors would have to be separated and more classes/.py files created.

##### Change 4: kept .yaml, moved configurable constants (vars, pressure lvls, etc)

Configuration options are largely still left to the YAML files (see `example_yaml_files` directory for up-to-date examples). Other configuration options, such as variable names, stats to calculate, etc have been moved to `model/constants.py`. This makes adding more stats types, pressure levels, or variable names easy in the future. For example, the default code only calculates `['f', 'acorr', 'rms']` stats for global stats, which can help speed up code a lot.

## Repository Structure

### Quickstart

The current code is split into smaller files so that each file has a narrower job than the original single-file v1 workflow. You do not need to understand every file before making a useful change. A good first pass is:

1. Start with `sbatch_stats.run` to see how the workflow is launched on Discover.
2. Look at `stats.py` and `controller/cli_controller.py` to see the high-level flow of the code.
3. Look to edit files under `model/` if there is a scientific or data-processing behavior you want to change.

These are the main set of files used in a stats workflow run (indented files represent code called by previous code):

```text
sbatch_stats.run
  -> stats.py
    -> controller/cli_controller.py
      -> model/dataset_parallel_executor.py
        -> model/dataset_processor.py
          -> model/dataset_regridder.py
      -> model/statistics_parallel_executor.py
        -> model/statistics_processor.py
```

The archived code (separated into versions v1/v2/v3) is kept under `archives/legacy_versions` for comparison, but they do not affect the current workflow.

**Note:** you will often see the statistics workflow referred to as a "pipeline", this is a term to define  all of the steps the code takes from running the command with a .YAML file to getting stats output files.

### Repo structure Diagram

```text
weather_fm_stats/
  stats.py                           # starts the current Python pipeline
  sbatch_stats.run                   # normal Discover SLURM submit script
  salloc_stats.run                   # run inside an existing SLURM allocation
  compare_v1_v4_runtimes.run         # compare archived v1 against current code

  controller/
    cli_controller.py                # reads CLI (command-line) options and runs pipeline phases
    dataset_controller.py            # controls the high-level dataset (fcst/ana/clim) operations
    stats_controller.py              # controls the high-level stats (reg/glo) operations
    merge_controller.py              # controls the high-level merge operations

  model/
    dataset_processor.py             # finds files, validates variables, builds datasets
    dataset_regridder.py             # regrids fields onto the target grid
    dataset_parallel_executor.py     # runs fcst/ana/clim dataset chunks in parallel
    statistics_processor.py          # computes regional and global statistics
    statistics_parallel_executor.py  # runs regional/global statistics chunks in parallel
    config_model.py                  # reads YAML settings and runtime defaults
    worker_controls.py               # chooses worker counts from YAML and SLURM limits
    chunk_plan.py                    # splits dates into deterministic chunks
    constants.py                     # variable names, aliases, regions, statistic names

  view/
    console_view.py                  # small console-printing helpers

  example_yaml_files/
    short_exp/                       # short examples for quick testing
    long_exp/                        # longer May 2024 examples

  tests/
    python/                          # fast regression tests
    shell/                           # SLURM wrappers for tests on Discover

  archives/
    legacy_versions/                 # old v1/v2/v3 code for comparison only
```

For most changes, this is the easiest place to start:

| If you want to... | Start here |
| :--- | :--- |
| Change YAML options, defaults, or worker settings | `model/config_model.py`, `model/worker_controls.py` |
| Change how raw files are found, variables are validated, or datasets are assembled | `model/dataset_processor.py` |
| Change regridding onto the target grid | `model/dataset_regridder.py`, then `model/dataset_processor.py` |
| Change how forecast/analysis/climatology chunks run in parallel | `model/dataset_parallel_executor.py`, `model/chunk_plan.py` |
| Change regional or global statistics calculations | `model/statistics_processor.py` |
| Change statistics chunking or statistics worker counts | `model/statistics_parallel_executor.py` |
| Change the overall pipeline order, phase summaries, or CLI flags | `controller/cli_controller.py` |
| Change SLURM submission behavior | `sbatch_stats.run`, `salloc_stats.run` |
| Add or update example YAML files | `example_yaml_files/short_exp`, `example_yaml_files/long_exp` |
| Add regression tests | `tests/python` |

The main idea is that science-specific behavior usually lives in `model/`, while workflow wiring lives in `controller/` and the root `.run` scripts. If you are unsure where a behavior lives, search for the printed log message or variable name you recognize from a run log.

## Workflow

**Note: Workflow runs exclusively on the Discover cluster. The workflow requires a .yaml file to describe what experiment configuration you'd like to run. See `example_yaml_files` for formatting.**

The normal workflow is a single SLURM job with in-process parallelism for dataset and statistics chunks. Use `sbatch_stats.run` from a login node. Lower-level Python flags are intended for debugging, merge recovery, and manual reruns.

Pipeline outputs are written to `outputs/` at the repository root. Expected outputs will look like:

```text
  outputs/
  |-- stats_{fcst_model}_{ana_model}_{clim_model}_{start_date}-{end_date}_{timestamp}/
  |   |-- jobs/        # copied YAML config, generated SLURM scripts, helper scripts
  |   |-- logs/        # SLURM/Python pipeline logs
  |   |-- tmp/         # temporary chunk outputs used during processing
  |   `-- run_summary.txt
  |-- fcst_{fcst_model}_{start_date}-{end_date}_len{fcst_length}d_int{fcst_interval}h_spc{fcst_spacing}d_{Nlat}x{Nlon}.nc4
  |-- ana_{ana_model}_{start_date}-{end_date}_len{fcst_length}d_int{fcst_interval}h_spc{fcst_spacing}d_{Nlat}x{Nlon}.nc4
  |-- clim_{clim_model}_{start_date}-{end_date}_len{fcst_length}d_int{fcst_interval}h_spc{fcst_spacing}d_{Nlat}x{Nlon}.nc4
  |-- stats_regional_{fcst_model}_{ana_model}_{clim_model}_{start_date}-{end_date}_len{fcst_length}d_int{fcst_interval}h_spc{fcst_spacing}d_{Nlat}x{Nlon}.nc4
  `-- stats_global_{fcst_model}_{ana_model}_{clim_model}_{start_date}-{end_date}_len{fcst_length}d_int{fcst_interval}h_spc{fcst_spacing}d_{Nlat}x{Nlon}.nc4
```

The `stats_..._{timestamp}/` directory is run-specific metadata and scratch space. The root-level `.nc4` files are the reusable forecast, analysis,
climatology, regional statistics, and global statistics products.

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

Supported ML models are GenCast, AIFS, Prithvi. Supported reanalysis/climatology models are GEOSFP, MERRA2, and ERA5.

**Note: different models will have different file globbing/date organization patterns. To have these loaded correctly, look at the example .yaml files for help.**

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

## Regression Testing

Fast Python regression tests live under `tests/python`. These tests should use synthetic data or small mocked inputs where possible, so they can check core logic without requiring SLURM jobs, Discover-only data paths, or full end-to-end workflow runs.

Run Python regression tests directly with:

```bash
python -m pytest tests/python
```

On Discover/HPC, run the Python regression tests through SLURM with:

```bash
chmod +x tests/shell/sbatch_python_tests.run
./tests/shell/sbatch_python_tests.run
```

The SLURM wrapper loads the Python module stack, sets `PYTHONPATH` to the repository root, runs every Python test file under `tests/python`, and writes job scripts/logs under `tests/logs`.

Common wrapper overrides:

```bash
TEST_TIME=01:00:00 TEST_MEM=32G TEST_CPUS=8 ./tests/shell/sbatch_python_tests.run
```

Use `compare_v1_v4_runtimes.run` separately for full workflow regression checks against the archived v1 implementation. That script submits short and long v1/current jobs, checks expected outputs, and prints timing comparisons. It is useful for periodic validation, but it is slower and more sensitive to SLURM scheduling than the Python regression tests.

## Comparison between original code and current pipeline

Below is an example run comparing long and short experiments for the archived original `v1` code and the current single-job pipeline. There are slight timing differences between each run, due to SLURM scheduling. This experiment was run on 2026-07-09. In this run, the current pipeline was faster for both example workflows while keeping the cleaner single-job structure.

| Experiment | v1 elapsed | current elapsed | current-v1 |
| :---: | :---: | :---: | :---: |
| short | 00:10:40 | 00:04:00 | -00:06:40 |
| long | 00:27:11 | 00:19:01 | -00:08:10 |
