'''CLI controller for selecting and running stats workflows.'''

import argparse
from contextlib import contextmanager
import os
import sys
import time
import traceback
from datetime import datetime

from model.config_model import resolve_runtime_settings
from model.dataset_parallel_executor import run_parallel_source_dataset_build
from model.dataset_processor import BatchDatasetProcessor
from model.statistics_parallel_executor import (
    get_current_memory_mb,
    run_parallel_statistics_branch,
)
from model.statistics_processor import StatisticsProcessor

# ================== MAIN FUNCTION ==================


def _read_current_rss_mb():
    '''Return current process RSS in MB on Linux when available.'''
    try:
        with open('/proc/self/statm', 'r', encoding='utf-8') as statm_file:
            rss_pages = int(statm_file.read().split()[1])
        page_size = os.sysconf('SC_PAGE_SIZE')
        return round((rss_pages * page_size) / (1024 * 1024), 2)
    except Exception:
        return None


def _read_max_rss_mb(include_children=False):
    '''Return max RSS in MB from resource.getrusage when available.'''
    try:
        import resource
        usage_target = (resource.RUSAGE_CHILDREN if include_children
                        else resource.RUSAGE_SELF)
        max_rss = resource.getrusage(usage_target).ru_maxrss
    except Exception:
        return None

    if sys.platform == 'darwin':
        return round(max_rss / (1024 * 1024), 2)
    return round(max_rss / 1024, 2)


def _cpu_total_seconds():
    '''Return self + completed child CPU seconds.'''
    times = os.times()
    return round(
        times.user + times.system + times.children_user +
        times.children_system,
        4,
    )


def _allocated_cpu_count():
    '''Return SLURM CPU allocation when present, otherwise local CPU count.'''
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return os.cpu_count() or 1


def _slurm_memory_mb():
    '''Return configured SLURM memory if exported by the scheduler.'''
    mem_per_node = os.environ.get('SLURM_MEM_PER_NODE')
    mem_per_cpu = os.environ.get('SLURM_MEM_PER_CPU')
    cpus = _allocated_cpu_count()
    try:
        if mem_per_node:
            return int(mem_per_node)
        if mem_per_cpu:
            return int(mem_per_cpu) * cpus
    except ValueError:
        return None
    return None


@contextmanager
def record_pipeline_stage(stage_metrics, stage_name):
    '''Record wall time, CPU time, and memory around one pipeline stage.'''
    start_wall = time.time()
    start_cpu = _cpu_total_seconds()
    start_current_rss = _read_current_rss_mb()
    start_max_rss = _read_max_rss_mb()
    start_child_max_rss = _read_max_rss_mb(include_children=True)
    cpu_count = _allocated_cpu_count()

    print(f'[RESOURCE] START {stage_name}: '
          f'current_rss_mb={start_current_rss} '
          f'max_rss_mb={start_max_rss} cpus={cpu_count} '
          f'slurm_mem_mb={_slurm_memory_mb()}')
    try:
        yield
    finally:
        end_wall = time.time()
        end_cpu = _cpu_total_seconds()
        wall_seconds = round(end_wall - start_wall, 2)
        cpu_seconds = round(end_cpu - start_cpu, 2)
        cpu_percent_of_allocation = None
        if wall_seconds > 0 and cpu_count:
            cpu_percent_of_allocation = round(
                (cpu_seconds / (wall_seconds * cpu_count)) * 100, 2)
        end_current_rss = _read_current_rss_mb()
        end_max_rss = _read_max_rss_mb()
        end_child_max_rss = _read_max_rss_mb(include_children=True)
        max_rss_values = [
            value for value in [start_max_rss, end_max_rss,
                                start_child_max_rss, end_child_max_rss]
            if value is not None
        ]
        max_rss_mb = max(max_rss_values) if max_rss_values else None
        record = {
            'stage': stage_name,
            'wall_seconds': wall_seconds,
            'cpu_seconds': cpu_seconds,
            'cpu_percent_of_allocation': cpu_percent_of_allocation,
            'allocated_cpus': cpu_count,
            'start_current_rss_mb': start_current_rss,
            'end_current_rss_mb': end_current_rss,
            'max_rss_mb': max_rss_mb,
            'self_max_rss_mb': end_max_rss,
            'children_max_rss_mb': end_child_max_rss,
            'slurm_mem_mb': _slurm_memory_mb(),
        }
        stage_metrics.append(record)
        print(f'[RESOURCE] END {stage_name}: '
              f'wall_seconds={wall_seconds} cpu_seconds={cpu_seconds} '
              f'cpu_pct_of_alloc={cpu_percent_of_allocation} '
              f'end_current_rss_mb={end_current_rss} '
              f'max_rss_mb={max_rss_mb}')


def parse_arguments():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='stats.yaml',
                        help='Configuration file (default: stats.yaml)')
    parser.add_argument('--date', type=str,
                        help=('Date YYYYMMDD for single-forecast mode '
                              '(process all F/A/C datasets and calculate'
                              'regional stats and scorecard input)'))
    parser.add_argument('--save_dir', type=str,
                        help=('Directory to save output files (for single '
                              'forecast mode)'))
    parser.add_argument('--pipeline', action='store_true',
                        help='Run end-to-end single-job pipeline orchestration')
    parser.add_argument('--stats_types',
                        choices=['regional', 'global', 'both'],
                        help='Pipeline stats branch selection')
    parser.add_argument('--pipeline_fail_policy',
                        choices=['fail_fast', 'partial_ok'],
                        help='Pipeline failure handling policy')
    parser.add_argument('--pipeline_branch_execution',
                        choices=['parallel', 'sequential'],
                        help='Pipeline branch execution mode')
    parser.add_argument('--pipeline_resume_mode',
                        choices=['off', 'safe'],
                        help='Pipeline resume mode')
    parser.add_argument('--pipeline_summary_file',
                        help='Pipeline summary filename in output directory')
    parser.add_argument('--pipeline_max_workers_dataset', type=int,
                        help='Maximum worker processes for dataset build stage')
    parser.add_argument('--pipeline_chunk_size_fcst', type=int,
                        help='Forecast init-date chunk size for dataset parallelism')
    parser.add_argument('--pipeline_chunk_size_ana', type=int,
                        help='Analysis valid-time chunk size for dataset parallelism')
    parser.add_argument('--pipeline_chunk_size_clim', type=int,
                        help='Climatology valid-time chunk size for dataset parallelism')
    parser.add_argument('--pipeline_max_workers_stats', type=int,
                        help='Maximum worker processes for statistics stage')
    parser.add_argument('--pipeline_chunk_size_stats', type=int,
                        help='Statistics init-date chunk size')
    # Processing options
    process_group = parser.add_argument_group('Processing options')
    process_group.add_argument('--check_only', action='store_true',
                               help='Only check for pre-existing datasets')
    process_group.add_argument('--fcst', action='store_true',
                               help='Process only forecast dataset')
    process_group.add_argument('--ana', action='store_true',
                               help='Process only analysis dataset')
    process_group.add_argument('--clim', action='store_true',
                               help='Process only climatology dataset')
    process_group.add_argument('--process', action='store_true',
                               help='Process all datasets (fcst, ana, clim)')
    process_group.add_argument('--collection', dest='target_coll',
                               default=None,
                               help=('Process only specific collection (e.g., '
                                     'default, slices, aerosol)'))
    process_group.add_argument('--info_dir', default=None,
                               help='Directory with date range and timestamp')
    process_group.add_argument('--date_start_idx', type=int,
                               help='Starting index for processing date range')
    process_group.add_argument('--date_end_idx', type=int,
                               help='Ending index for processing date range')

    # Statistics options
    stats_group = parser.add_argument_group('Statistics options')
    stats_group.add_argument('--stats', choices=['reg', 'glo'],
                             help=('Run statistics calculation type: reg or '
                                   'glo'))
    stats_group.add_argument('--init_start_idx', type=int,
                             help='Starting index for init dates (for chunks)')
    stats_group.add_argument('--init_end_idx', type=int,
                             help='Ending index for init dates (for chunks)')
    stats_group.add_argument('--chunk', type=int,
                             help='Forecast chunk number')
    stats_group.add_argument('--chunk_size', type=int, default=3,
                             help='Forecast chunk size (default: 3)')

    # Merge options
    merge_group = parser.add_argument_group('Merge options')
    merge_group.add_argument('--merge_collections',
                             choices=['fcst', 'ana', 'clim'],
                             help=('Dataset to merge collections (fcst, ana, '
                             'or clim)'))
    merge_group.add_argument('--merge_forecast_chunks', action='store_true',
                             help='Merge forecast chunk files')
    merge_group.add_argument('--save_for_coll_merge', action='store_true',
                             help=('Save merged fcst file to tmp for '
                                   'collection merging'))
    merge_group.add_argument('--clean', action='store_true',
                             help='Merge chunked statistics files')
    merge_group.add_argument('--type', choices=['reg', 'glo'],
                             help='stats type to merge: reg or glo')

    return parser.parse_args()


def get_explicit_cli_args(argv):
    '''Return command-line option names that were explicitly provided.'''
    argv_set = set()
    i = 1
    while i < len(argv):
        if argv[i].startswith('--'):
            arg_name = argv[i][2:]  # Remove '--'
            argv_set.add(arg_name)
        i += 1
    return argv_set


def validate_single_forecast_args(args, single_fcst_mode):
    '''Validate arguments for single-forecast mode.'''
    if not single_fcst_mode:
        return

    if not args.save_dir:
        print('[ERROR] --save_dir is required when using --date')
        sys.exit(1)
    try:
        datetime.strptime(single_fcst_mode, '%Y%m%d')
    except ValueError:
        print('[ERROR] --date must be in YYYYMMDD format')
        sys.exit(1)

    # Only check arguments that were explicitly provided.
    incompatible = [
        arg for arg in get_explicit_cli_args(sys.argv)
        if arg not in ['date', 'save_dir', 'config']
    ]
    if incompatible:
        print('[ERROR] Single-forecast mode (implied by --date) only '
              'accepts --date, --save_dir, and --config')
        print(f'Found incompatible arguments: {incompatible}')
        sys.exit(1)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


def ensure_regular_output_dir(single_fcst_mode):
    '''Ensure the standard output directory exists outside single-forecast mode.'''
    if not single_fcst_mode and not os.path.exists('outputs'):
        os.makedirs('outputs')


def apply_default_processing_mode(args):
    '''Default to full dataset processing when no explicit mode is selected.'''
    if not (args.pipeline or args.fcst or args.ana or args.clim or args.process or args.stats or
        args.merge_collections or args.merge_forecast_chunks or args.clean):
        args.process = True


def validate_runtime_args(args, single_fcst_mode):
    '''Validate regular processing, statistics, and merge argument combinations.'''
    if args.pipeline:
        if args.stats:
            print('[ERROR] --pipeline cannot be combined with --stats. '
                  'Use --stats_types for pipeline branch selection.')
            sys.exit(1)
        if (args.fcst or args.ana or args.clim or args.check_only or
            args.target_coll or args.init_start_idx is not None or
            args.init_end_idx is not None or args.chunk is not None or
            args.date_start_idx is not None or args.date_end_idx is not None):
            print('[ERROR] --pipeline cannot be combined with dataset or '
                  'chunk-specific flags')
            sys.exit(1)
        if args.merge_collections or args.merge_forecast_chunks or args.clean:
            print('[ERROR] --pipeline cannot be combined with merge-only flags')
            sys.exit(1)
        if not args.info_dir and not single_fcst_mode:
            print('[ERROR] --info_dir is required when using --pipeline')
            sys.exit(1)

    if not args.info_dir and not single_fcst_mode and (
            args.check_only or args.init_start_idx is not None or
            args.init_end_idx is not None or args.clean or args.target_coll or
            args.date_start_idx is not None or args.date_end_idx is not None or
            args.merge_collections is not None or args.merge_forecast_chunks):
        print('[ERROR] --info_dir is required for --check_only, chunked '
              'processing, --clean, --collection, --merge_collections, or'
              '--merge_forecast_chunks')
        sys.exit(1)

    using_chunks = (args.init_start_idx is not None or
                    args.init_end_idx is not None)
    if using_chunks:
        if args.init_start_idx is None or args.init_end_idx is None:
            print('[ERROR] When using chunked processing, both '
                  '--init-start-idx and --init-end-idx are required')
            sys.exit(1)
        if args.chunk is None:
            print('[ERROR] --chunk is required when using chunked processing')
            sys.exit(1)

    processing_flags = [args.process, args.fcst, args.ana, args.clim]
    if sum(processing_flags) > 1:
        print('[ERROR] Only one of --process, --fcst, --ana, --clim can '
              'be specified')
        sys.exit(1)

    if args.stats and any([args.fcst, args.ana, args.clim]):
        print('[ERROR] --stats cannot be used with individual processing '
              'flags (--fcst, --ana, --clim)')
        sys.exit(1)

    return using_chunks


def prepare_runtime(args, single_fcst_mode):
    '''Apply defaults and validate command-line state before processing.'''
    validate_single_forecast_args(args, single_fcst_mode)
    ensure_regular_output_dir(single_fcst_mode)
    apply_default_processing_mode(args)
    return validate_runtime_args(args, single_fcst_mode)


def run_merge_command(args):
    '''Run merge-only commands. Return True when a command was handled.'''
    if args.merge_collections:
        if args.merge_collections not in ['fcst', 'ana', 'clim']:
            print('[ERROR] --merge_collections must be fcst, ana, or clim')
            sys.exit(1)
        processor = BatchDatasetProcessor.from_yaml(args.config)
        success = processor.merge_collection_datasets(
            args.merge_collections, args.info_dir)
        if success:
            print('\n[SUCCESS] Collection datasets merged successfully')
            return True
        print('\n[ERROR] Failed to merge collection datasets')
        traceback.print_exc()
        sys.exit(1)

    if args.merge_forecast_chunks:
        processor = BatchDatasetProcessor.from_yaml(args.config)
        success = processor.merge_forecast_chunks(args.info_dir,
                                                  args.save_for_coll_merge)
        if success:
            print('\n[SUCCESS] Forecast chunks merged successfully')
            return True
        print('\n[ERROR] Failed to merge forecast chunks')
        traceback.print_exc()
        sys.exit(1)

    if args.clean:
        if args.type not in ['reg', 'glo']:
            print('[ERROR] --typer must be reg or glo')
            sys.exit(1)
        success = StatisticsProcessor.merge_statistics_files(
            args.type, args.info_dir)
        if success:
            print('\n[SUCCESS] Statistics files merged successfully')
            return True
        print('\n[ERROR] Failed to merge statistics files')
        traceback.print_exc()
        sys.exit(1)

    return False


def print_runtime_contract(runtime_settings):
    '''Print the resolved step-1 runtime contract settings.'''
    print('\nResolved pipeline runtime contract:')
    print(f'  stats_types={runtime_settings.stats_types}')
    print(f'  fail_policy={runtime_settings.pipeline_fail_policy}')
    print(f'  branch_execution={runtime_settings.pipeline_branch_execution}')
    print(f'  resume_mode={runtime_settings.pipeline_resume_mode}')
    print(f'  summary_file={runtime_settings.pipeline_summary_file}')
    print(f'  max_workers_dataset={runtime_settings.pipeline_max_workers_dataset}')
    print(f'  chunk_size_fcst={runtime_settings.pipeline_chunk_size_fcst}')
    print(f'  chunk_size_ana={runtime_settings.pipeline_chunk_size_ana}')
    print(f'  chunk_size_clim={runtime_settings.pipeline_chunk_size_clim}')
    print(f'  max_workers_stats={runtime_settings.pipeline_max_workers_stats}')
    print(f'  chunk_size_stats={runtime_settings.pipeline_chunk_size_stats}')


def run_stats_branch(stats_kind, processor_config, dataset_files, init_dates,
                     leads, info_dir, runtime_settings=None):
    '''Run one statistics branch and return status/error.'''
    if runtime_settings is not None:
        return run_parallel_statistics_branch(
            stats_kind, processor_config, dataset_files, init_dates, leads,
            info_dir, runtime_settings)

    try:
        stats_processor = StatisticsProcessor(
            processor_config, dataset_files=dataset_files,
            init_dates=init_dates, leads=leads)
        if stats_kind == 'regional':
            stats_processor.run_regional_statistics(
                info_dir=info_dir, using_chunks=False, chunk_number=None,
                skip_avg=False)
        elif stats_kind == 'global':
            stats_processor.run_global_statistics(
                info_dir=info_dir, using_chunks=False, chunk_number=None,
                skip_avg=False)
        else:
            raise ValueError(f'Unsupported stats branch: {stats_kind}')
        return {'status': 'SUCCESS', 'error': ''}
    except Exception as exc:
        traceback.print_exc()
        return {'status': 'FAILURE', 'error': str(exc)}


def write_pipeline_summary(info_dir, runtime_settings, branch_results,
                           final_status, stage_metrics=None):
    '''Write run summary file for pipeline mode.'''
    if not info_dir:
        return
    summary_dir = os.path.join('outputs', info_dir)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir,
                                runtime_settings.pipeline_summary_file)
    pipeline_max_memory = None
    stage_metrics = stage_metrics or []
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('v4 pipeline summary\n')
        f.write(f'final_status: {final_status}\n')
        f.write(f'stats_types: {runtime_settings.stats_types}\n')
        f.write(f'pipeline_fail_policy: '
                f'{runtime_settings.pipeline_fail_policy}\n')
        f.write(f'pipeline_branch_execution: '
                f'{runtime_settings.pipeline_branch_execution}\n')
        f.write(f'pipeline_resume_mode: '
                f'{runtime_settings.pipeline_resume_mode}\n')
        for branch_name, branch_result in branch_results.items():
            f.write(f'{branch_name}_status: {branch_result["status"]}\n')
            if branch_result['error']:
                f.write(f'{branch_name}_error: {branch_result["error"]}\n')
            if 'chunk_count' in branch_result:
                f.write(f'{branch_name}_chunk_count: '
                        f'{branch_result["chunk_count"]}\n')
                f.write(f'{branch_name}_required_chunk_count: '
                        f'{branch_result["required_chunk_count"]}\n')
                f.write(f'{branch_name}_skipped_chunk_count: '
                        f'{branch_result["skipped_chunk_count"]}\n')
            if 'max_workers' in branch_result:
                f.write(f'{branch_name}_max_workers: '
                        f'{branch_result["max_workers"]}\n')
            if branch_result.get('max_memory_mb') is not None:
                f.write(f'{branch_name}_max_memory_mb: '
                        f'{branch_result["max_memory_mb"]}\n')
        branch_memory = [
            result.get('max_memory_mb') for result in branch_results.values()
            if result.get('max_memory_mb') is not None
        ]
        current_memory = get_current_memory_mb()
        memory_values = branch_memory + (
            [current_memory] if current_memory is not None else [])
        if memory_values:
            pipeline_max_memory = max(memory_values)
            f.write(f'pipeline_max_memory_mb: {pipeline_max_memory}\n')
        if stage_metrics:
            f.write('\nstage_resource_summary:\n')
            for metric in stage_metrics:
                prefix = metric['stage'].replace(' ', '_').lower()
                f.write(f'{prefix}_wall_seconds: '
                        f'{metric["wall_seconds"]}\n')
                f.write(f'{prefix}_cpu_seconds: '
                        f'{metric["cpu_seconds"]}\n')
                f.write(f'{prefix}_cpu_percent_of_allocation: '
                        f'{metric["cpu_percent_of_allocation"]}\n')
                f.write(f'{prefix}_allocated_cpus: '
                        f'{metric["allocated_cpus"]}\n')
                f.write(f'{prefix}_start_current_rss_mb: '
                        f'{metric["start_current_rss_mb"]}\n')
                f.write(f'{prefix}_end_current_rss_mb: '
                        f'{metric["end_current_rss_mb"]}\n')
                f.write(f'{prefix}_max_rss_mb: '
                        f'{metric["max_rss_mb"]}\n')
                f.write(f'{prefix}_self_max_rss_mb: '
                        f'{metric["self_max_rss_mb"]}\n')
                f.write(f'{prefix}_children_max_rss_mb: '
                        f'{metric["children_max_rss_mb"]}\n')
                f.write(f'{prefix}_slurm_mem_mb: '
                        f'{metric["slurm_mem_mb"]}\n')
    print(f'[INFO] Pipeline summary written: {summary_path}')
    print('[INFO] Pipeline final summary:')
    for branch_name, branch_result in branch_results.items():
        print(f'  {branch_name}: {branch_result["status"]}')
        if 'chunk_count' in branch_result:
            print(f'    chunks={branch_result["chunk_count"]} '
                  f'required={branch_result["required_chunk_count"]} '
                  f'skipped={branch_result["skipped_chunk_count"]}')
        if 'max_workers' in branch_result:
            print(f'    max_workers={branch_result["max_workers"]}')
        if branch_result.get('max_memory_mb') is not None:
            print(f'    max_memory_mb={branch_result["max_memory_mb"]}')
    if pipeline_max_memory is not None:
        print(f'  pipeline_max_memory_mb={pipeline_max_memory}')
    if stage_metrics:
        print('  stage resource usage:')
        for metric in stage_metrics:
            print(f'    {metric["stage"]}: '
                  f'wall={metric["wall_seconds"]}s '
                  f'cpu={metric["cpu_seconds"]}s '
                  f'cpu_pct_alloc={metric["cpu_percent_of_allocation"]} '
                  f'max_rss_mb={metric["max_rss_mb"]}')


def run_pipeline_mode(args, single_fcst_mode):
    '''Run end-to-end pipeline orchestration using one controller path.'''
    print('\n==================================================')
    print('V4 SINGLE-JOB PIPELINE')
    print('==================================================')
    stage_metrics = []
    print('[INFO] Stage 1/4: Preflight')

    with record_pipeline_stage(stage_metrics, 'stage_1_preflight'):
        processor = BatchDatasetProcessor.from_yaml(args.config,
                                                    single_fcst_mode)
        runtime_settings = resolve_runtime_settings(args, processor.config)
        print_runtime_contract(runtime_settings)
        if runtime_settings.pipeline_resume_mode == 'safe':
            print('[INFO] Resume mode is safe '
                  '(full checkpointing pending later step)')

    print('[INFO] Stage 2/4: Source dataset build')
    with record_pipeline_stage(stage_metrics, 'stage_2_source_dataset_build'):
        results = run_parallel_source_dataset_build(
            args.config,
            args.info_dir,
            runtime_settings,
            single_fcst_mode=single_fcst_mode,
        )

    if results.get('status') != 'success':
        reason = results.get('reason', 'Unknown error')
        print(f'[ERROR] Dataset creation failed in pipeline mode: {reason}')
        for err in results.get('errors', []):
            print(f'  [ERROR] {err}')
        write_pipeline_summary(args.info_dir, runtime_settings, {},
                               'FAILURE', stage_metrics=stage_metrics)
        return 1

    dataset_files = results.get('dataset_files', {})
    init_dates = results.get('init_dates', [])
    leads = results.get('leads', [])

    required_datasets = {'fcst', 'ana', 'clim'}
    missing_datasets = required_datasets - set(dataset_files.keys())
    if missing_datasets:
        print(f'[ERROR] Missing datasets for stats: '
              f'{", ".join(sorted(missing_datasets))}')
        write_pipeline_summary(args.info_dir, runtime_settings, {},
                               'FAILURE', stage_metrics=stage_metrics)
        return 1
    if not init_dates or not leads:
        print('[ERROR] Missing init dates or lead times for statistics')
        write_pipeline_summary(args.info_dir, runtime_settings, {},
                               'FAILURE', stage_metrics=stage_metrics)
        return 1
    print(f'[INFO] RESULTS FROM STAGE 2 (datasets): {dataset_files}')

    print('[INFO] Stage 3/4: Statistics branches')
    requested_branches = []
    if runtime_settings.stats_types in ['regional', 'both']:
        requested_branches.append('regional')
    if runtime_settings.stats_types in ['global', 'both']:
        requested_branches.append('global')

    if (runtime_settings.pipeline_branch_execution == 'parallel' and
        len(requested_branches) > 1):
        print('[INFO] Branch execution=parallel configured; running sequentially '
              'for now in step 2/3 orchestration.')

    branch_results = {}
    fail_fast_triggered = False
    with record_pipeline_stage(stage_metrics, 'stage_3_statistics_branches'):
        for branch in requested_branches:
            print(f'[INFO] Running {branch} statistics branch...')
            branch_results[branch] = run_stats_branch(
                branch, processor.config, dataset_files, init_dates, leads,
                args.info_dir, runtime_settings=runtime_settings)
            if (branch_results[branch]['status'] != 'SUCCESS' and
                runtime_settings.pipeline_fail_policy == 'fail_fast'):
                print('[ERROR] fail_fast policy triggered')
                fail_fast_triggered = True
                break
    if fail_fast_triggered:
        write_pipeline_summary(args.info_dir, runtime_settings,
                               branch_results, 'FAILURE',
                               stage_metrics=stage_metrics)
        return 1

    success_count = sum(1 for r in branch_results.values()
                        if r['status'] == 'SUCCESS')
    if success_count == len(requested_branches):
        final_status = 'SUCCESS'
        exit_code = 0
    elif success_count == 0:
        final_status = 'FAILURE'
        exit_code = 1
    else:
        final_status = 'PARTIAL_FAILURE'
        exit_code = 1

    print('[INFO] Stage 4/4: Finalize')
    with record_pipeline_stage(stage_metrics, 'stage_4_finalize'):
        pass
    write_pipeline_summary(args.info_dir, runtime_settings, branch_results,
                           final_status, stage_metrics=stage_metrics)
    if final_status == 'SUCCESS':
        print('\n[SUCCESS] Pipeline completed successfully')
    elif final_status == 'PARTIAL_FAILURE':
        print('\n[ERROR] Pipeline finished with partial failure')
    else:
        print('\n[ERROR] Pipeline failed')
    return exit_code


def main():
    '''Main function'''

    # Parse arguments
    args = parse_arguments()
    single_fcst_mode = args.date

    using_chunks = prepare_runtime(args, single_fcst_mode)
    skip_calc_mode = bool(args.target_coll)

    try:
        if args.pipeline:
            exit_code = run_pipeline_mode(args, single_fcst_mode)
            sys.exit(exit_code)

        if run_merge_command(args):
            return

        # Phase 1: Dataset Processing
        dataset_files = {}
        init_dates = []
        leads = []

        # Determine which datasets to process
        process_fcst = args.process or args.fcst
        process_ana = args.process or args.ana
        process_clim = args.process or args.clim

        # Proceed with dataset processing if not stats-only
        # (except in single-forecast mode which needs both)
        if not (args.stats and not args.process) or args.date:

            print('\n==================================================')
            if args.stats:
                print('DATASET CREATION AND STATISTICS CALCULATION')
            else:
                print('DATASET CREATION')
            print('==================================================')

            processor = BatchDatasetProcessor.from_yaml(args.config,
                                                        single_fcst_mode)

            # Process batch configured with command line arguments
            if not process_fcst:
                processor.fcst_model = ''
            if not process_ana:
                processor.ana_model  = ''
            if not process_clim:
                processor.clim_model = ''
            results = processor.process_batch(
                target_coll=args.target_coll, info_dir=args.info_dir,
                date_start_idx=args.date_start_idx,
                date_end_idx=args.date_end_idx, check_only=args.check_only,
                skip_calc_mode=skip_calc_mode,
                single_fcst_mode=single_fcst_mode)

            # Exit if dataset processing does not result in a type of success
            if results['status'] == 'check_only_success':
                sys.exit(0)
            elif results['status'] != 'success':
                print(f'[ERROR] Dataset creation failed: '
                      f'{results.get("reason", "Unknown error")}')
                sys.exit(1)

            # Save results for statistics calculation
            dataset_files = results.get('dataset_files', {})
            init_dates = results.get('init_dates', [])
            leads = results.get('leads', [])

            processor.save_processed_datasets(
                results,
                info_dir=args.info_dir,
                date_start_idx=args.date_start_idx,
                date_end_idx=args.date_end_idx,
                target_coll=args.target_coll,
                single_fcst_mode=bool(single_fcst_mode),
                skip_calc_mode=skip_calc_mode,
            )

        # Phase 2: Statistics Calculation
        if args.stats or args.date:  # also run in single-fcst mode
            print('\n==================================================')
            print('STATISTICS CALCULATION')
            print('==================================================')

            # Initialize needed variables if in stats-only mode
            if not dataset_files or not init_dates or not leads:
                processor = BatchDatasetProcessor.from_yaml(args.config)
                spacing = processor.config.get('fcst_spacing', 1)
                exclude_dates = processor.config.get('exclude_dates', [])
                init_dates = processor._parse_date_range_with_spacing(
                    processor.config['FDATES'], spacing, exclude_dates)
                leads = processor._generate_leads(
                    processor.config['FDAYS'], processor.config['NFREQ'])
                existing_datasets = processor._check_for_existing_datasets(
                    stats_only_mode=True)
                for dataset_type, file_path in existing_datasets.items():
                    dataset_files[dataset_type] = file_path

            # Check if all datasets and neeeded variables are present
            required_datasets = {'fcst', 'ana', 'clim'}
            available_datasets = set(dataset_files.keys())
            missing_datasets = required_datasets - available_datasets
            if missing_datasets:
                print(f'[ERROR] Missing datasets for statistics calculation: '
                      f'{", ".join(missing_datasets)}')
                print('[INFO] Statistics requires forecast, analysis, and '
                      'climatology datasets')
                sys.exit(1)
            if not init_dates or not leads:
                print('[ERROR] Missing init dates or lead times - cannot '
                      'calculate statistics')
                sys.exit(1)

            # Create statistics processor
            if single_fcst_mode:
                # Single-forecast mode: use in-memory datasets
                stats_processor = StatisticsProcessor(
                    processor.config, in_memory_datasets=results['datasets'],
                    init_dates=init_dates, leads=leads)
            else:
                # Regular mode: use file paths
                stats_processor = StatisticsProcessor(
                    processor.config, dataset_files=dataset_files,
                    init_dates=init_dates, leads=leads)

            # Handle chunking if specified
            if using_chunks:
                start_idx = max(0, min(args.init_start_idx, len(init_dates)-1))
                end_idx = max(0, min(args.init_end_idx, len(init_dates)))
                # Graceful exit if chunk is empty
                chunked_dates = init_dates[start_idx:end_idx]
                if len(chunked_dates) == 0:
                    print('[INFO] This chunk contains no valid dates')
                    print('[SUCCESS] No processing needed for this chunk')
                    sys.exit(0)
                stats_processor.init_dates = init_dates[start_idx:end_idx]
                print(f'[INFO] Processing init dates from index {start_idx} '
                      f'to {end_idx}')
                init_dates_formatted = [d.strftime('%Y-%m-%d')
                                        for d in stats_processor.init_dates]
                print(f'[INFO] Selected {len(stats_processor.init_dates)} '
                      f'init dates: {init_dates_formatted}')

            # Run appropriate statistics
            if args.stats == 'reg' or args.date: # also for single-fcst mode
                stats_processor.run_regional_statistics(
                    info_dir=args.info_dir,
                    using_chunks=using_chunks,
                    chunk_number=args.chunk if using_chunks else None,
                    skip_avg=using_chunks or args.date,
                    save_dir=args.save_dir if args.date else None)
            elif args.stats == 'glo':
                stats_processor.run_global_statistics(
                    info_dir=args.info_dir,
                    using_chunks=using_chunks,
                    chunk_number=args.chunk if using_chunks else None,
                    skip_avg=using_chunks)

    # Print success or error
        print('\n[SUCCESS] All requested operations completed!')
    except FileNotFoundError as e:
        print(f'\n[ERROR] File not found: {e}')
        traceback.print_exc()
        sys.exit(1)
    except ValueError as e:
        print(f'\n[ERROR] Configuration error: {e}')
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f'\n[ERROR] Unexpected error: {e}')
        traceback.print_exc()
        sys.exit(1)




