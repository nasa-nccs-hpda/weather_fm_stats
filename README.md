# Repository description
Agentic development of scientific weather model stats.

## Workflow
Workflow runs exclusively on the Discover cluster. The workflow requires a .yaml file to describe what experiment configuration you'd like to run. See `example_yaml_files` for formatting.

You can either run this code from a Discover login node, or directly from an allocated A100 gpu node. 

### Option 1: from login node

```bash
cd path/to/repo/v4
chmod +x sbatch_stats_v4.run
./sbatch_stats_v4.run <yaml_filename>
```

### Option 2: from GPU node

```bash
salloc --job-name=stats --time=24:00:00 --gres=gpu:1 --mem 200G -c 10 -p gpu_a100 --constraint="rome"
cd path/to/repo/v4
chmod +x salloc_run_v4.sh
./salloc_stats_v4.sh <yaml filename>
```


