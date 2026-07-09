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
from model.worker_controls import get_slurm_cpu_cap


PIPELINE_RUN_CONTEXT = {}

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
def record_pipeline_phase(phase_metrics, phase_name):
    '''Record wall time, CPU time, and memory around one pipeline phase.'''
    start_wall = time.time()
    start_cpu = _cpu_total_seconds()
    start_current_rss = _read_current_rss_mb()
    start_max_rss = _read_max_rss_mb()
    start_child_max_rss = _read_max_rss_mb(include_children=True)
    cpu_count = _allocated_cpu_count()

    print(f'[RESOURCE] START {phase_name}: '
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
            'phase': phase_name,
            'stage': phase_name,
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
        phase_metrics.append(record)
        print(f'[RESOURCE] END {phase_name}: '
              f'wall_seconds={wall_seconds} cpu_seconds={cpu_seconds} '
              f'cpu_pct_of_alloc={cpu_percent_of_allocation} '
              f'end_current_rss_mb={end_current_rss} '
              f'max_rss_mb={max_rss_mb}')


def parse_arguments():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(
        description=(
            'Run the v4 weather statistics pipeline. The normal v4 path is '
            '--pipeline, usually launched by sbatch_stats_v4.run or '
            'salloc_stats_v4.run. Lower-level dataset, statistics, and merge '
            'flags are kept for debugging, recovery, and manual reruns.'
        ),
    )
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
                        help=('Run the end-to-end single-job v4 pipeline '
                              '(normal production path)'))
    parser.add_argument('--stats_types',
                        choices=['regional', 'global', 'both'],
                        help='Pipeline stats branch selection')
    parser.add_argument('--pipeline_fail_policy',
                        choices=['fail_fast', 'partial_ok'],
                        help='Pipeline failure handling policy')
    parser.add_argument('--pipeline_branch_execution',
                        choices=['sequential'],
                        help=('Pipeline branch execution mode '
                              '(v4 supports sequential regional/global '
                              'branches only)'))
    parser.add_argument('--pipeline_resume_mode',
                        choices=['off', 'safe'],
                        help='Pipeline resume mode')
    parser.add_argument('--pipeline_summary_file',
                        help='Pipeline summary filename in output directory')
    parser.add_argument('--pipeline_log_level',
                        choices=['normal', 'verbose', 'debug'],
                        help='Pipeline logging verbosity')
    parser.add_argument('--pipeline_max_workers_dataset', type=int,
                        help='Maximum worker processes for dataset build phase')
    parser.add_argument('--pipeline_max_workers_dataset_fcst', type=int,
                        help='Maximum worker processes for forecast dataset chunks')
    parser.add_argument('--pipeline_max_workers_dataset_ana', type=int,
                        help='Maximum worker processes for analysis dataset chunks')
    parser.add_argument('--pipeline_max_workers_dataset_clim', type=int,
                        help='Maximum worker processes for climatology dataset chunks')
    parser.add_argument('--pipeline_chunk_size_fcst', type=int,
                        help='Forecast init-date chunk size for dataset parallelism')
    parser.add_argument('--pipeline_chunk_size_ana', type=int,
                        help='Analysis valid-time chunk size for dataset parallelism')
    parser.add_argument('--pipeline_chunk_size_clim', type=int,
                        help='Climatology valid-time chunk size for dataset parallelism')
    parser.add_argument('--pipeline_max_workers_stats', type=int,
                        help='Maximum worker processes for statistics stage')
    parser.add_argument('--pipeline_max_workers_stats_regional', type=int,
                        help='Maximum worker processes for regional statistics')
    parser.add_argument('--pipeline_max_workers_stats_global', type=int,
                        help='Maximum worker processes for global statistics')
    parser.add_argument('--pipeline_chunk_size_stats', type=int,
                        help='Statistics init-date chunk size')
    # Manual/debug processing options
    process_group = parser.add_argument_group(
        'Manual/debug dataset options')
    process_group.add_argument('--check_only', action='store_true',
                               help='Only check for pre-existing datasets')
    process_group.add_argument('--fcst', action='store_true',
                               help='Manually process only forecast dataset')
    process_group.add_argument('--ana', action='store_true',
                               help='Manually process only analysis dataset')
    process_group.add_argument('--clim', action='store_true',
                               help='Manually process only climatology dataset')
    process_group.add_argument('--process', action='store_true',
                               help=('Manually process all datasets '
                                     '(fcst, ana, clim)'))
    process_group.add_argument('--collection', dest='target_coll',
                               default=None,
                               help=('Process only specific collection (e.g., '
                                     'default, slices, aerosol)'))
    process_group.add_argument('--info_dir', default=None,
                               help=('Run output directory name used for '
                                     'manual/debug recovery commands'))
    process_group.add_argument('--date_start_idx', type=int,
                               help='Starting index for processing date range')
    process_group.add_argument('--date_end_idx', type=int,
                               help='Ending index for processing date range')

    # Manual/debug statistics options
    stats_group = parser.add_argument_group(
        'Manual/debug statistics options')
    stats_group.add_argument('--stats', choices=['reg', 'glo'],
                             help=('Manually run statistics calculation type: '
                                   'reg or glo'))
    stats_group.add_argument('--init_start_idx', type=int,
                             help='Starting index for init dates (for chunks)')
    stats_group.add_argument('--init_end_idx', type=int,
                             help='Ending index for init dates (for chunks)')
    stats_group.add_argument('--chunk', type=int,
                             help='Forecast chunk number')
    stats_group.add_argument('--chunk_size', type=int, default=3,
                             help='Forecast chunk size (default: 3)')

    # Manual/debug merge options
    merge_group = parser.add_argument_group('Manual/debug merge options')
    merge_group.add_argument('--merge_collections',
                             choices=['fcst', 'ana', 'clim'],
                             help=('Dataset to merge collections (fcst, ana, '
                             'or clim)'))
    merge_group.add_argument('--merge_forecast_chunks', action='store_true',
                             help='Manually merge forecast chunk files')
    merge_group.add_argument('--save_for_coll_merge', action='store_true',
                             help=('Save merged fcst file to tmp for '
                                   'collection merging'))
    merge_group.add_argument('--clean', action='store_true',
                             help='Manually merge chunked statistics files')
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
    if runtime_settings.pipeline_log_level == 'normal':
        return

    print('\nResolved pipeline runtime contract:')
    print(f'  stats_types={runtime_settings.stats_types}')
    print(f'  fail_policy={runtime_settings.pipeline_fail_policy}')
    print(f'  branch_execution={runtime_settings.pipeline_branch_execution}')
    print(f'  resume_mode={runtime_settings.pipeline_resume_mode}')
    print(f'  summary_file={runtime_settings.pipeline_summary_file}')
    print(f'  log_level={runtime_settings.pipeline_log_level}')
    print(f'  max_workers_dataset={runtime_settings.pipeline_max_workers_dataset}')
    print(f'  max_workers_dataset_fcst={runtime_settings.pipeline_max_workers_dataset_fcst}')
    print(f'  max_workers_dataset_ana={runtime_settings.pipeline_max_workers_dataset_ana}')
    print(f'  max_workers_dataset_clim={runtime_settings.pipeline_max_workers_dataset_clim}')
    print(f'  chunk_size_fcst={runtime_settings.pipeline_chunk_size_fcst}')
    print(f'  chunk_size_ana={runtime_settings.pipeline_chunk_size_ana}')
    print(f'  chunk_size_clim={runtime_settings.pipeline_chunk_size_clim}')
    print(f'  max_workers_stats={runtime_settings.pipeline_max_workers_stats}')
    print(f'  max_workers_stats_regional={runtime_settings.pipeline_max_workers_stats_regional}')
    print(f'  max_workers_stats_global={runtime_settings.pipeline_max_workers_stats_global}')
    print(f'  slurm_cpu_cap={get_slurm_cpu_cap()}')
    print(f'  chunk_size_stats={runtime_settings.pipeline_chunk_size_stats}')


def print_pipeline_phase_header(phase_number, total_phases, title):
    '''Print a consistent ASCII pipeline phase separator.'''
    line = '=' * 72
    print(f'\n{line}')
    print(f'PHASE {phase_number}/{total_phases}: {title}')
    print(line)


def print_pipeline_phase_footer():
    '''Print a separator after one pipeline phase completes.'''
    print('=' * 80)


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


def finalize_branch_result(branch_name, branch_result, branch_start_time,
                           branch_start_label, branch_order,
                           requested_branch_count):
    '''Add pipeline-level branch metadata for summaries.'''
    branch_result.setdefault('branch', branch_name)
    branch_result.setdefault('error', '')
    branch_result['execution_mode'] = 'sequential'
    branch_result['branch_order'] = branch_order
    branch_result['requested_branch_count'] = requested_branch_count
    branch_result['start_time'] = branch_start_label
    branch_result['end_time'] = datetime.now().isoformat()
    branch_result['wall_seconds'] = round(time.time() - branch_start_time, 2)
    worker_cpu_seconds = branch_result.get('worker_cpu_seconds')
    if worker_cpu_seconds is not None:
        allocated_cpus = _allocated_cpu_count()
        branch_result['worker_cpu_percent_of_allocation'] = None
        if branch_result['wall_seconds'] > 0 and allocated_cpus:
            branch_result['worker_cpu_percent_of_allocation'] = round(
                (worker_cpu_seconds /
                 (branch_result['wall_seconds'] * allocated_cpus)) * 100, 2)
        branch_result['allocated_cpus'] = allocated_cpus
    output_path = branch_result.get('output_path')
    if output_path:
        branch_result['output_exists'] = os.path.exists(output_path)
    return branch_result


def skipped_branch_result(branch_name, reason, branch_order,
                          requested_branch_count):
    '''Create a summary record for a branch skipped by policy.'''
    return {
        'status': 'SKIPPED',
        'error': reason,
        'branch': branch_name,
        'execution_mode': 'sequential',
        'branch_order': branch_order,
        'requested_branch_count': requested_branch_count,
        'wall_seconds': 0.0,
    }


def _summary_separator(title=None):
    '''Print an 80-character summary separator with an optional title.'''
    line = '=' * 80
    print(line)
    if title:
        print(title)
        print(line)


def _fmt_seconds(value):
    '''Format seconds for human-readable summaries.'''
    if value is None:
        return 'n/a'
    return f'{value}s'


def _fmt_value(value):
    '''Format possibly missing summary values.'''
    return 'n/a' if value is None else str(value)


def _dataset_timing_groups(dataset_timings):
    '''Group dataset timing records by pipeline dataset prefix.'''
    groups = []
    group_map = {}
    for timing in dataset_timings:
        name = timing['name']
        if name.startswith('fcst_'):
            group_name = 'fcst'
        elif name.startswith('ana_'):
            group_name = 'ana'
        elif name.startswith('clim_'):
            group_name = 'clim'
        elif name.startswith('source_'):
            group_name = 'source'
        else:
            group_name = 'other'
        if group_name not in group_map:
            group_map[group_name] = []
            groups.append((group_name, group_map[group_name]))
        group_map[group_name].append(timing)
    return groups


def _print_resource_summary(phase_metrics, dataset_timings, branch_results,
                            pipeline_max_memory):
    '''Print hierarchical resource usage summary.'''
    _summary_separator('RESOURCE USAGE')
    if pipeline_max_memory is not None:
        print(f'  pipeline max RSS: {pipeline_max_memory} MB')
    print('  phases:')
    for metric in phase_metrics:
        print(f'    {metric["phase"]}: '
              f'max_rss_mb={_fmt_value(metric["max_rss_mb"])} '
              f'cpu_seconds={_fmt_value(metric["cpu_seconds"])} '
              f'cpu_pct_alloc='
              f'{_fmt_value(metric["cpu_percent_of_allocation"])}')
    if dataset_timings:
        print('  dataset sub-steps:')
        for group_name, timings in _dataset_timing_groups(dataset_timings):
            print(f'    {group_name}:')
            for timing in timings:
                print(f'      {timing["name"]}: '
                      f'cpu_seconds={_fmt_value(timing.get("cpu_seconds"))} '
                      f'cpu_pct_alloc='
                      f'{_fmt_value(timing.get("cpu_percent_of_allocation"))} '
                      f'allocated_cpus='
                      f'{_fmt_value(timing.get("allocated_cpus"))}')
    if branch_results:
        print('  statistics branches:')
        for branch_name, result in branch_results.items():
            print(f'    {branch_name}: '
                  f'max_memory_mb='
                  f'{_fmt_value(result.get("max_memory_mb"))} '
                  f'worker_cpu_seconds='
                  f'{_fmt_value(result.get("worker_cpu_seconds"))} '
                  f'worker_cpu_pct_alloc='
                  f'{_fmt_value(result.get("worker_cpu_percent_of_allocation"))} '
                  f'max_workers={_fmt_value(result.get("max_workers"))} '
                  f'configured_workers='
                  f'{_fmt_value(result.get("configured_max_workers"))} '
                  f'slurm_cpu_cap={_fmt_value(result.get("slurm_cpu_cap"))}')


def _print_timing_summary(phase_metrics, dataset_timings, branch_results,
                          pipeline_wall_seconds, dataset_statuses=None):
    '''Print hierarchical timing summary.'''
    dataset_statuses = dataset_statuses or {}
    _summary_separator('TIMING')
    print(f'  overall python pipeline: {_fmt_seconds(pipeline_wall_seconds)}')
    print('  phases:')
    for metric in phase_metrics:
        print(f'    {metric["phase"]}: '
              f'wall={_fmt_seconds(metric["wall_seconds"])}')
    if dataset_timings:
        print('  dataset sub-steps:')
        for group_name, timings in _dataset_timing_groups(dataset_timings):
            if group_name in dataset_statuses:
                print(f'    {group_name}: status={dataset_statuses[group_name]}')
            else:
                print(f'    {group_name}:')
            for timing in timings:
                print(f'      {timing["name"]}: '
                      f'wall={_fmt_seconds(timing["wall_seconds"])}')
    if branch_results:
        print('  statistics branches:')
        for branch_name, result in branch_results.items():
            print(f'    {branch_name}: '
                  f'wall={_fmt_seconds(result.get("wall_seconds"))} '
                  f'status={result["status"]}')


def _print_output_summary(branch_results, dataset_statuses=None,
                          dataset_files=None):
    '''Print branch/output status summary.'''
    dataset_statuses = dataset_statuses or {}
    dataset_files = dataset_files or {}
    _summary_separator('OUTPUTS AND STATUS')
    if dataset_statuses:
        print('  source datasets:')
        for dataset_type in ['fcst', 'ana', 'clim']:
            if dataset_type in dataset_statuses:
                print(f'    {dataset_type}: {dataset_statuses[dataset_type]}')
                if dataset_files.get(dataset_type):
                    print(f'      output={dataset_files[dataset_type]}')
    if branch_results:
        print('  statistics branches:')
    for branch_name, branch_result in branch_results.items():
        print(f'    {branch_name}: {branch_result["status"]}')
        if branch_result.get('error'):
            print(f'      error={branch_result["error"]}')
        if branch_result.get('reuse_reason'):
            print(f'      reuse_reason={branch_result["reuse_reason"]}')
        if 'chunk_count' in branch_result:
            print(f'      chunks={branch_result["chunk_count"]} '
                  f'required={branch_result["required_chunk_count"]} '
                  f'skipped={branch_result["skipped_chunk_count"]} '
                  f'completed={branch_result.get("completed_chunk_count")} '
                  f'failed={branch_result.get("failed_chunk_count")}')
        if branch_result.get('output_path'):
            print(f'      output={branch_result["output_path"]} '
                  f'exists={branch_result.get("output_exists")}')


def write_pipeline_summary(info_dir, runtime_settings, branch_results,
                           final_status, phase_metrics=None,
                           dataset_timings=None,
                           pipeline_wall_seconds=None,
                           dataset_statuses=None,
                           dataset_files=None):
    '''Write run summary file for pipeline mode.'''
    if not info_dir:
        return
    summary_dir = os.path.join('outputs', info_dir)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir,
                                runtime_settings.pipeline_summary_file)
    pipeline_max_memory = None
    phase_metrics = phase_metrics or []
    dataset_timings = dataset_timings or []
    dataset_statuses = dataset_statuses or {}
    dataset_files = dataset_files or {}
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('v4 pipeline summary\n')
        f.write(f'final_status: {final_status}\n')
        f.write(f'pipeline_wall_seconds: {pipeline_wall_seconds}\n')
        f.write(f'stats_types: {runtime_settings.stats_types}\n')
        f.write(f'pipeline_fail_policy: '
                f'{runtime_settings.pipeline_fail_policy}\n')
        f.write(f'pipeline_branch_execution: '
                f'{runtime_settings.pipeline_branch_execution}\n')
        f.write(f'pipeline_resume_mode: '
                f'{runtime_settings.pipeline_resume_mode}\n')
        f.write(f'pipeline_log_level: '
                f'{runtime_settings.pipeline_log_level}\n')
        if dataset_statuses:
            f.write('\nsource_dataset_statuses:\n')
            for dataset_type in ['fcst', 'ana', 'clim']:
                if dataset_type in dataset_statuses:
                    f.write(f'{dataset_type}_status: '
                            f'{dataset_statuses[dataset_type]}\n')
                    if dataset_files.get(dataset_type):
                        f.write(f'{dataset_type}_output_path: '
                                f'{dataset_files[dataset_type]}\n')
        for branch_name, branch_result in branch_results.items():
            f.write(f'{branch_name}_status: {branch_result["status"]}\n')
            if branch_result.get('error'):
                f.write(f'{branch_name}_error: {branch_result["error"]}\n')
            if branch_result.get('reuse_reason'):
                f.write(f'{branch_name}_reuse_reason: '
                        f'{branch_result["reuse_reason"]}\n')
            if branch_result.get('execution_mode'):
                f.write(f'{branch_name}_execution_mode: '
                        f'{branch_result["execution_mode"]}\n')
            if branch_result.get('branch_order') is not None:
                f.write(f'{branch_name}_branch_order: '
                        f'{branch_result["branch_order"]}\n')
            if branch_result.get('start_time'):
                f.write(f'{branch_name}_start_time: '
                        f'{branch_result["start_time"]}\n')
            if branch_result.get('end_time'):
                f.write(f'{branch_name}_end_time: '
                        f'{branch_result["end_time"]}\n')
            if 'chunk_count' in branch_result:
                f.write(f'{branch_name}_chunk_count: '
                        f'{branch_result["chunk_count"]}\n')
                f.write(f'{branch_name}_required_chunk_count: '
                        f'{branch_result["required_chunk_count"]}\n')
                f.write(f'{branch_name}_skipped_chunk_count: '
                        f'{branch_result["skipped_chunk_count"]}\n')
                f.write(f'{branch_name}_completed_chunk_count: '
                        f'{branch_result.get("completed_chunk_count")}\n')
                f.write(f'{branch_name}_failed_chunk_count: '
                        f'{branch_result.get("failed_chunk_count")}\n')
            if 'chunk_size' in branch_result:
                f.write(f'{branch_name}_chunk_size: '
                        f'{branch_result["chunk_size"]}\n')
            if 'max_workers' in branch_result:
                f.write(f'{branch_name}_max_workers: '
                        f'{branch_result["max_workers"]}\n')
            if 'configured_max_workers' in branch_result:
                f.write(f'{branch_name}_configured_max_workers: '
                        f'{branch_result["configured_max_workers"]}\n')
            if branch_result.get('slurm_cpu_cap') is not None:
                f.write(f'{branch_name}_slurm_cpu_cap: '
                        f'{branch_result["slurm_cpu_cap"]}\n')
            if branch_result.get('max_memory_mb') is not None:
                f.write(f'{branch_name}_max_memory_mb: '
                        f'{branch_result["max_memory_mb"]}\n')
            if branch_result.get('wall_seconds') is not None:
                f.write(f'{branch_name}_wall_seconds: '
                        f'{branch_result["wall_seconds"]}\n')
            if branch_result.get('worker_cpu_seconds') is not None:
                f.write(f'{branch_name}_worker_cpu_seconds: '
                        f'{branch_result["worker_cpu_seconds"]}\n')
                f.write(f'{branch_name}_worker_cpu_percent_of_allocation: '
                        f'{branch_result.get("worker_cpu_percent_of_allocation")}\n')
                f.write(f'{branch_name}_allocated_cpus: '
                        f'{branch_result.get("allocated_cpus")}\n')
            if branch_result.get('output_path'):
                f.write(f'{branch_name}_output_path: '
                        f'{branch_result["output_path"]}\n')
                f.write(f'{branch_name}_output_exists: '
                        f'{branch_result.get("output_exists")}\n')
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
        if phase_metrics:
            f.write('\nphase_resource_summary:\n')
            for metric in phase_metrics:
                prefix = metric['phase'].replace(' ', '_').lower()
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
        if dataset_timings:
            f.write('\ndataset_timing_summary:\n')
            for timing in dataset_timings:
                f.write(f'{timing["name"]}_wall_seconds: '
                        f'{timing["wall_seconds"]}\n')
                f.write(f'{timing["name"]}_cpu_seconds: '
                        f'{timing.get("cpu_seconds")}\n')
                f.write(f'{timing["name"]}_cpu_percent_of_allocation: '
                        f'{timing.get("cpu_percent_of_allocation")}\n')
                f.write(f'{timing["name"]}_allocated_cpus: '
                        f'{timing.get("allocated_cpus")}\n')
        f.write('\n' + '=' * 80 + '\n')
        f.write('RESOURCE USAGE\n')
        f.write('=' * 80 + '\n')
        f.write(f'pipeline_max_memory_mb: {pipeline_max_memory}\n')
        f.write('phases:\n')
        for metric in phase_metrics:
            f.write(f'  {metric["phase"]}: '
                    f'max_rss_mb={metric["max_rss_mb"]} '
                    f'cpu_seconds={metric["cpu_seconds"]} '
                    f'cpu_pct_alloc='
                    f'{metric["cpu_percent_of_allocation"]}\n')
        if dataset_timings:
            f.write('dataset sub-steps:\n')
            for group_name, timings in _dataset_timing_groups(dataset_timings):
                f.write(f'  {group_name}:\n')
                for timing in timings:
                    f.write(f'    {timing["name"]}: '
                            f'cpu_seconds={timing.get("cpu_seconds")} '
                            f'cpu_pct_alloc='
                            f'{timing.get("cpu_percent_of_allocation")} '
                            f'allocated_cpus='
                            f'{timing.get("allocated_cpus")}\n')
        if branch_results:
            f.write('statistics branches:\n')
            for branch_name, result in branch_results.items():
                f.write(f'  {branch_name}: '
                        f'max_memory_mb={result.get("max_memory_mb")} '
                        f'worker_cpu_seconds='
                        f'{result.get("worker_cpu_seconds")} '
                        f'worker_cpu_pct_alloc='
                        f'{result.get("worker_cpu_percent_of_allocation")} '
                        f'max_workers={result.get("max_workers")} '
                        f'configured_workers='
                        f'{result.get("configured_max_workers")} '
                        f'slurm_cpu_cap={result.get("slurm_cpu_cap")}\n')
        f.write('\n' + '=' * 80 + '\n')
        f.write('TIMING\n')
        f.write('=' * 80 + '\n')
        f.write(f'overall_python_pipeline: '
                f'{pipeline_wall_seconds}s\n')
        f.write('phases:\n')
        for metric in phase_metrics:
            f.write(f'  {metric["phase"]}: '
                    f'wall={metric["wall_seconds"]}s\n')
        if dataset_timings:
            f.write('dataset sub-steps:\n')
            for group_name, timings in _dataset_timing_groups(dataset_timings):
                if group_name in dataset_statuses:
                    f.write(f'  {group_name}: '
                            f'status={dataset_statuses[group_name]}\n')
                else:
                    f.write(f'  {group_name}:\n')
                for timing in timings:
                    f.write(f'    {timing["name"]}: '
                            f'wall={timing["wall_seconds"]}s\n')
        if branch_results:
            f.write('statistics branches:\n')
            for branch_name, result in branch_results.items():
                f.write(f'  {branch_name}: '
                        f'wall={result.get("wall_seconds")}s '
                        f'status={result["status"]}\n')
    print(f'[INFO] Pipeline summary written: {summary_path}')
    _summary_separator('PIPELINE FINAL SUMMARY')
    print(f'  final_status: {final_status}')
    print(f'  pipeline_wall_seconds: {_fmt_seconds(pipeline_wall_seconds)}')
    print(f'  stats_types: {runtime_settings.stats_types}')
    print(f'  resume_mode: {runtime_settings.pipeline_resume_mode}')
    if dataset_statuses:
        print('  source datasets:')
        for dataset_type in ['fcst', 'ana', 'clim']:
            if dataset_type in dataset_statuses:
                print(f'    {dataset_type}: {dataset_statuses[dataset_type]}')
                if dataset_files.get(dataset_type):
                    print(f'      output={dataset_files[dataset_type]}')
    _print_output_summary(branch_results, dataset_statuses, dataset_files)
    _print_resource_summary(phase_metrics, dataset_timings, branch_results,
                            pipeline_max_memory)
    _print_timing_summary(phase_metrics, dataset_timings, branch_results,
                          pipeline_wall_seconds, dataset_statuses)


def write_pipeline_exception_summary(args, exc):
    '''Write a best-effort failure summary after an uncaught pipeline error.'''
    try:
        context = PIPELINE_RUN_CONTEXT
        runtime_settings = context.get('runtime_settings')
        if not runtime_settings or not getattr(args, 'info_dir', None):
            return False
        pipeline_start = context.get('pipeline_start')
        pipeline_wall_seconds = None
        if pipeline_start is not None:
            pipeline_wall_seconds = round(time.time() - pipeline_start, 2)
        branch_results = context.get('branch_results') or {}
        if branch_results:
            branch_results = dict(branch_results)
        branch_results['_pipeline_exception'] = {
            'status': 'FAILURE',
            'error': str(exc),
            'branch': '_pipeline_exception',
            'execution_mode': 'pipeline',
            'wall_seconds': 0.0,
        }
        write_pipeline_summary(
            args.info_dir,
            runtime_settings,
            branch_results,
            'FAILURE',
            phase_metrics=context.get('phase_metrics') or [],
            dataset_timings=context.get('dataset_timings') or [],
            pipeline_wall_seconds=pipeline_wall_seconds,
            dataset_statuses=context.get('dataset_statuses') or {},
            dataset_files=context.get('dataset_files') or {},
        )
        return True
    except Exception as summary_exc:
        print(f'[WARNING] Could not write pipeline failure summary: '
              f'{summary_exc}')
        return False


def run_pipeline_mode(args, single_fcst_mode):
    '''Run end-to-end pipeline orchestration using one controller path.'''
    print('\n==================================================')
    print('V4 SINGLE-JOB PIPELINE')
    print('==================================================')
    pipeline_start = time.time()
    phase_metrics = []
    dataset_timings = []
    branch_results = {}
    PIPELINE_RUN_CONTEXT.clear()
    PIPELINE_RUN_CONTEXT.update({
        'pipeline_start': pipeline_start,
        'phase_metrics': phase_metrics,
        'dataset_timings': dataset_timings,
        'branch_results': branch_results,
        'dataset_statuses': {},
        'dataset_files': {},
        'runtime_settings': None,
    })
    print_pipeline_phase_header(1, 4, 'Preflight')

    with record_pipeline_phase(phase_metrics, 'phase_1_preflight'):
        processor = BatchDatasetProcessor.from_yaml(args.config,
                                                    single_fcst_mode)
        runtime_settings = resolve_runtime_settings(args, processor.config)
        PIPELINE_RUN_CONTEXT['runtime_settings'] = runtime_settings
        print_runtime_contract(runtime_settings)
        if runtime_settings.pipeline_resume_mode == 'safe':
            print('[INFO] Resume mode is safe '
                  '(validated final outputs may be reused; chunk-level '
                  'checkpointing is not enabled)')
    print_pipeline_phase_footer()

    print_pipeline_phase_header(2, 4, 'Source dataset build')
    with record_pipeline_phase(phase_metrics, 'phase_2_source_dataset_build'):
        results = run_parallel_source_dataset_build(
            args.config,
            args.info_dir,
            runtime_settings,
            single_fcst_mode=single_fcst_mode,
        )
    print_pipeline_phase_footer()
    dataset_timings = results.get('timings', [])
    PIPELINE_RUN_CONTEXT['dataset_timings'] = dataset_timings
    dataset_statuses = results.get('dataset_statuses', {})
    PIPELINE_RUN_CONTEXT['dataset_statuses'] = dataset_statuses

    if results.get('status') != 'success':
        reason = results.get('reason', 'Unknown error')
        print(f'[ERROR] Dataset creation failed in pipeline mode: {reason}')
        for err in results.get('errors', []):
            print(f'  [ERROR] {err}')
        write_pipeline_summary(args.info_dir, runtime_settings, {},
                               'FAILURE', phase_metrics=phase_metrics,
                               dataset_timings=dataset_timings,
                               pipeline_wall_seconds=round(
                                   time.time() - pipeline_start, 2),
                               dataset_statuses=dataset_statuses,
                               dataset_files=results.get('dataset_files', {}))
        return 1

    dataset_files = results.get('dataset_files', {})
    PIPELINE_RUN_CONTEXT['dataset_files'] = dataset_files
    init_dates = results.get('init_dates', [])
    leads = results.get('leads', [])

    required_datasets = {'fcst', 'ana', 'clim'}
    missing_datasets = required_datasets - set(dataset_files.keys())
    if missing_datasets:
        print(f'[ERROR] Missing datasets for stats: '
              f'{", ".join(sorted(missing_datasets))}')
        write_pipeline_summary(args.info_dir, runtime_settings, {},
                               'FAILURE', phase_metrics=phase_metrics,
                               dataset_timings=dataset_timings,
                               pipeline_wall_seconds=round(
                                   time.time() - pipeline_start, 2),
                               dataset_statuses=dataset_statuses,
                               dataset_files=dataset_files)
        return 1
    if not init_dates or not leads:
        print('[ERROR] Missing init dates or lead times for statistics')
        write_pipeline_summary(args.info_dir, runtime_settings, {},
                               'FAILURE', phase_metrics=phase_metrics,
                               dataset_timings=dataset_timings,
                               pipeline_wall_seconds=round(
                                   time.time() - pipeline_start, 2),
                               dataset_statuses=dataset_statuses,
                               dataset_files=dataset_files)
        return 1
    if dataset_statuses:
        print('[INFO] Source dataset statuses:')
        for dataset_type in ['fcst', 'ana', 'clim']:
            if dataset_type in dataset_statuses:
                output_path = dataset_files.get(dataset_type, 'n/a')
                print(f'  {dataset_type}: {dataset_statuses[dataset_type]} '
                      f'output={output_path}')

    print_pipeline_phase_header(3, 4, 'Statistics branches')
    requested_branches = []
    if runtime_settings.stats_types in ['regional', 'both']:
        requested_branches.append('regional')
    if runtime_settings.stats_types in ['global', 'both']:
        requested_branches.append('global')

    print('[INFO] Statistics branch execution is sequential in v4')
    print(f'[INFO] Requested statistics branch order: '
          f'{", ".join(requested_branches)}')

    fail_fast_triggered = False
    with record_pipeline_phase(phase_metrics, 'phase_3_statistics_branches'):
        for branch_index, branch in enumerate(requested_branches, start=1):
            print(f'[INFO] Running {branch} statistics branch '
                  f'({branch_index}/{len(requested_branches)})...')
            print(f'[INFO] {branch} stats tuning: '
                  f'chunk_size={runtime_settings.pipeline_chunk_size_stats}, '
                  f'max_workers='
                  f'{runtime_settings.pipeline_max_workers_stats_regional if branch == "regional" else runtime_settings.pipeline_max_workers_stats_global}')
            branch_start = time.time()
            branch_start_label = datetime.now().isoformat()
            branch_result = run_stats_branch(
                branch, processor.config, dataset_files, init_dates, leads,
                args.info_dir, runtime_settings=runtime_settings)
            branch_results[branch] = finalize_branch_result(
                branch, branch_result, branch_start, branch_start_label,
                branch_index, len(requested_branches))
            print(f'[INFO] {branch} statistics branch finished: '
                  f'status={branch_results[branch]["status"]}, '
                  f'wall_seconds={branch_results[branch]["wall_seconds"]}')
            if (branch_results[branch]['status'] != 'SUCCESS' and
                branch_results[branch]['status'] != 'REUSED' and
                runtime_settings.pipeline_fail_policy == 'fail_fast'):
                print('[ERROR] fail_fast policy triggered')
                fail_fast_triggered = True
                for skipped_index in range(branch_index,
                                           len(requested_branches)):
                    skipped_branch = requested_branches[skipped_index]
                    if skipped_branch not in branch_results:
                        branch_results[skipped_branch] = skipped_branch_result(
                            skipped_branch,
                            'Skipped because fail_fast policy was triggered',
                            skipped_index + 1,
                            len(requested_branches))
                break
    print_pipeline_phase_footer()
    if fail_fast_triggered:
        write_pipeline_summary(args.info_dir, runtime_settings,
                               branch_results, 'FAILURE',
                               phase_metrics=phase_metrics,
                               dataset_timings=dataset_timings,
                               pipeline_wall_seconds=round(
                                   time.time() - pipeline_start, 2),
                               dataset_statuses=dataset_statuses,
                               dataset_files=dataset_files)
        return 1

    success_count = sum(1 for r in branch_results.values()
                        if r['status'] in ['SUCCESS', 'REUSED'])
    if success_count == len(requested_branches):
        final_status = 'SUCCESS'
        exit_code = 0
    elif success_count == 0:
        final_status = 'FAILURE'
        exit_code = 1
    else:
        final_status = 'PARTIAL_FAILURE'
        exit_code = 1

    print_pipeline_phase_header(4, 4, 'Finalize')
    with record_pipeline_phase(phase_metrics, 'phase_4_finalize'):
        pass
    print_pipeline_phase_footer()
    pipeline_wall_seconds = round(time.time() - pipeline_start, 2)
    write_pipeline_summary(args.info_dir, runtime_settings, branch_results,
                           final_status, phase_metrics=phase_metrics,
                           dataset_timings=dataset_timings,
                           pipeline_wall_seconds=pipeline_wall_seconds,
                           dataset_statuses=dataset_statuses,
                           dataset_files=dataset_files)
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

            print('\n  ==================================================')
            if args.stats:
                print('  DATASET CREATION AND STATISTICS CALCULATION')
            else:
                print('  DATASET CREATION')
            print('  ==================================================')

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
            print('\n  ==================================================')
            print('  STATISTICS CALCULATION')
            print('  ==================================================')

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
        write_pipeline_exception_summary(args, e)
        traceback.print_exc()
        sys.exit(1)
    except ValueError as e:
        print(f'\n[ERROR] Configuration error: {e}')
        write_pipeline_exception_summary(args, e)
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f'\n[ERROR] Unexpected error: {e}')
        write_pipeline_exception_summary(args, e)
        traceback.print_exc()
        sys.exit(1)




