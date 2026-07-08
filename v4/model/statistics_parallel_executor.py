'''Statistics parallel execution helpers for v4 pipeline mode.'''

import contextlib
import io
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import xarray as xr

from model.chunk_plan import InitDateChunkPlanner, write_chunk_plan
from model.statistics_processor import StatisticsProcessor


def _is_verbose_logging(log_level):
    '''Return True when detailed worker output should pass through.'''
    return str(log_level or 'normal').lower() in {'verbose', 'debug'}


def _run_with_chunk_output_capture(log_level, chunk_label, operation):
    '''Run a chunk operation while suppressing noisy internals by default.'''
    if _is_verbose_logging(log_level):
        return operation()

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            return operation()
    except Exception:
        captured_lines = [
            line for line in buffer.getvalue().splitlines()
            if line.strip()
        ]
        if captured_lines:
            print(f'[ERROR] {chunk_label} captured output before failure:')
            for line in captured_lines[-20:]:
                print(f'  {line}')
        raise


def get_current_memory_mb():
    '''Return current process max RSS in MB when available.'''
    try:
        import resource
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None

    if sys.platform == 'darwin':
        return round(max_rss / (1024 * 1024), 2)
    return round(max_rss / 1024, 2)


def _resolve_stats_workers(runtime_settings, num_chunks, stats_kind=None):
    '''Resolve stats worker count conservatively for memory-heavy work.'''
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        try:
            default_workers = max(1, int(slurm_cpus))
        except ValueError:
            default_workers = os.cpu_count() or 1
    else:
        default_workers = os.cpu_count() or 1

    configured_workers = runtime_settings.pipeline_max_workers_stats
    if stats_kind == 'regional':
        configured_workers = runtime_settings.pipeline_max_workers_stats_regional
    elif stats_kind == 'global':
        configured_workers = runtime_settings.pipeline_max_workers_stats_global
    if configured_workers:
        max_workers = configured_workers
    else:
        max_workers = min(default_workers, 4)
    return max(1, min(max_workers, num_chunks))


def _parse_chunk_dates(chunk_spec):
    '''Convert ChunkSpec selected date labels back to datetimes.'''
    return [
        datetime.strptime(date_value, '%Y-%m-%d')
        for date_value in chunk_spec.selected_dates
    ]


def _stats_type_for_branch(stats_kind):
    if stats_kind == 'regional':
        return 'reg'
    if stats_kind == 'global':
        return 'glo'
    raise ValueError(f'Unsupported stats branch: {stats_kind}')


def _expected_statistics_output_path(stats_kind, processor_config,
                                     dataset_files):
    '''Build expected final statistics output path for a branch.'''
    fcst_file = dataset_files.get('fcst')
    if not fcst_file:
        return None
    fcst_model = processor_config.get('fcst_model', '')
    ana_model = processor_config.get('ana_model', '')
    clim_model = processor_config.get('clim_model', '')
    try:
        suffix = os.path.basename(fcst_file).split(f'{fcst_model}_', 1)[1]
    except IndexError:
        return None
    if stats_kind == 'regional':
        return (f'outputs/stats_regional_{fcst_model}_{ana_model}_'
                f'{clim_model}_{suffix}')
    if stats_kind == 'global':
        return (f'outputs/stats_global_{fcst_model}_{ana_model}_'
                f'{clim_model}_{suffix}')
    return None


def _validate_existing_statistics_output(output_path, expected_init_dates,
                                         stats_kind):
    '''Lightly validate an existing merged statistics output.'''
    if not output_path or not os.path.exists(output_path):
        return False, 'output file does not exist'
    try:
        with xr.open_dataset(output_path, decode_timedelta=True) as ds:
            if 'init_date' not in ds.dims and 'init_date' not in ds.coords:
                return False, 'missing init_date dimension/coordinate'
            actual_init_dates = ds.sizes.get('init_date', len(ds.init_date))
            if actual_init_dates != expected_init_dates:
                return (
                    False,
                    f'init_date count {actual_init_dates} does not match '
                    f'expected {expected_init_dates}',
                )
            if stats_kind == 'regional' and 'region' not in ds.dims:
                return False, 'missing regional region dimension'
            if stats_kind == 'global':
                if 'lat' not in ds.dims or 'lon' not in ds.dims:
                    return False, 'missing global lat/lon dimensions'
        return True, ''
    except Exception as exc:
        return False, str(exc)


def _remove_stale_stats_chunks(chunk_specs):
    '''Remove old stats chunk outputs before rebuilding a branch.'''
    deleted_count = 0
    for chunk_spec in chunk_specs:
        if not chunk_spec.is_required:
            continue
        if os.path.exists(chunk_spec.output_path):
            try:
                os.remove(chunk_spec.output_path)
                deleted_count += 1
            except Exception as exc:
                print(f'[WARNING] Could not remove stale stats chunk '
                      f'{chunk_spec.output_path}: {exc}')
    if deleted_count:
        print(f'[INFO] Removed {deleted_count} stale statistics chunk file(s)')


def _run_stats_chunk_worker(stats_kind, processor_config, dataset_files,
                            leads, info_dir, chunk_spec, log_level='normal'):
    '''Worker entrypoint for one statistics chunk.'''
    chunk_label = f'{stats_kind} statistics chunk {chunk_spec.chunk_id}'

    def _run():
        stats_type = _stats_type_for_branch(stats_kind)
        stats_processor = StatisticsProcessor(
            processor_config,
            dataset_files=dataset_files,
            init_dates=_parse_chunk_dates(chunk_spec),
            leads=leads,
        )

        if stats_kind == 'regional':
            stats_processor.run_regional_statistics(
                info_dir=info_dir,
                using_chunks=True,
                chunk_number=chunk_spec.chunk_id,
                skip_avg=True,
            )
        else:
            stats_processor.run_global_statistics(
                info_dir=info_dir,
                using_chunks=True,
                chunk_number=chunk_spec.chunk_id,
                skip_avg=True,
            )

        return {
            'status': 'success',
            'chunk_index': chunk_spec.chunk_index,
            'chunk_id': chunk_spec.chunk_id,
            'start_idx': chunk_spec.start_idx,
            'end_idx': chunk_spec.end_idx,
            'output_path': chunk_spec.output_path,
            'processed_dates': len(chunk_spec.selected_dates),
            'max_memory_mb': get_current_memory_mb(),
            'stats_type': stats_type,
        }

    return _run_with_chunk_output_capture(log_level, chunk_label, _run)


def _chunk_result_from_spec(chunk_spec, status=None, error=None):
    result = {
        'status': status or chunk_spec.status,
        'chunk_index': chunk_spec.chunk_index,
        'chunk_id': chunk_spec.chunk_id,
        'start_idx': chunk_spec.start_idx,
        'end_idx': chunk_spec.end_idx,
        'output_path': chunk_spec.output_path,
        'processed_dates': len(chunk_spec.selected_dates),
        'max_memory_mb': get_current_memory_mb(),
    }
    if error:
        result['error'] = error
    return result


def run_parallel_statistics_branch(stats_kind, processor_config,
                                   dataset_files, init_dates, leads,
                                   info_dir, runtime_settings):
    '''Run one statistics branch with init-date chunk parallelism.'''
    stats_type = _stats_type_for_branch(stats_kind)
    expected_output_path = _expected_statistics_output_path(
        stats_kind, processor_config, dataset_files)
    configured_max_workers = (
        runtime_settings.pipeline_max_workers_stats_regional
        if stats_kind == 'regional'
        else runtime_settings.pipeline_max_workers_stats_global)

    if runtime_settings.pipeline_resume_mode == 'safe':
        is_valid, validation_error = _validate_existing_statistics_output(
            expected_output_path, len(init_dates), stats_kind)
        if is_valid:
            print(f'[INFO] Reusing existing {stats_kind} statistics output: '
                  f'{expected_output_path}')
            return {
                'status': 'REUSED',
                'error': '',
                'branch': stats_kind,
                'stats_type': stats_type,
                'chunk_results': [],
                'chunk_count': 0,
                'required_chunk_count': 0,
                'skipped_chunk_count': 0,
                'completed_chunk_count': 0,
                'failed_chunk_count': 0,
                'max_workers': 0,
                'configured_max_workers': configured_max_workers,
                'chunk_size': runtime_settings.pipeline_chunk_size_stats,
                'max_memory_mb': get_current_memory_mb(),
                'output_path': expected_output_path,
                'reuse_reason': 'valid existing statistics output',
            }
        if expected_output_path and os.path.exists(expected_output_path):
            print(f'[WARNING] Existing {stats_kind} statistics output will '
                  f'not be reused: {validation_error}')

    chunk_output_dir = os.path.join('outputs', str(info_dir), 'tmp')
    chunk_specs = InitDateChunkPlanner(
        init_dates,
        exclude_dates=[],
    ).build(
        runtime_settings.pipeline_chunk_size_stats,
        chunk_output_dir,
        f'stats_chunk_{stats_type}',
    )
    plan_path = os.path.join(
        chunk_output_dir, f'chunk_plan_stats_{stats_type}.json')
    write_chunk_plan(plan_path, chunk_specs)

    required_chunks = [
        chunk_spec for chunk_spec in chunk_specs
        if chunk_spec.is_required
    ]
    skipped_chunks = [
        chunk_spec for chunk_spec in chunk_specs
        if chunk_spec.is_skipped
    ]
    chunk_results = [
        _chunk_result_from_spec(chunk_spec) for chunk_spec in skipped_chunks
    ]

    if not required_chunks:
        return {
            'status': 'FAILURE',
            'error': 'No statistics init_dates available after exclusions',
            'branch': stats_kind,
            'stats_type': stats_type,
            'chunk_results': sorted(
                chunk_results, key=lambda item: item['chunk_index']),
            'chunk_count': len(chunk_specs),
            'required_chunk_count': 0,
            'skipped_chunk_count': len(skipped_chunks),
            'completed_chunk_count': 0,
            'failed_chunk_count': 0,
            'max_workers': 0,
            'configured_max_workers': configured_max_workers,
            'chunk_size': runtime_settings.pipeline_chunk_size_stats,
            'max_memory_mb': get_current_memory_mb(),
            'output_path': expected_output_path,
        }

    max_workers = _resolve_stats_workers(runtime_settings,
                                         len(required_chunks),
                                         stats_kind=stats_kind)
    print(f'[INFO] {stats_kind} stats chunking plan: '
          f'{len(chunk_specs)} chunk(s), {len(required_chunks)} required, '
          f'{len(skipped_chunks)} skipped, '
          f'chunk_size={runtime_settings.pipeline_chunk_size_stats}, '
          f'max_workers={max_workers}')
    _remove_stale_stats_chunks(required_chunks)

    chunk_failures = []
    completed_required_chunks = 0
    if max_workers == 1:
        for chunk_spec in required_chunks:
            chunk_start = time.time()
            print(f'[INFO] {stats_kind} stats chunk started: '
                  f'{chunk_spec.chunk_id} '
                  f'({completed_required_chunks + 1}/'
                  f'{len(required_chunks)})')
            try:
                result = _run_stats_chunk_worker(
                    stats_kind, processor_config, dataset_files, leads,
                    info_dir, chunk_spec,
                    log_level=runtime_settings.pipeline_log_level)
                chunk_spec.status = result['status']
                chunk_results.append(result)
                completed_required_chunks += 1
                print(f'[INFO] {stats_kind} stats chunk completed: '
                      f'{chunk_spec.chunk_id} '
                      f'progress={completed_required_chunks}/'
                      f'{len(required_chunks)} '
                      f'dates={result.get("processed_dates")} '
                      f'max_memory_mb={result.get("max_memory_mb")} '
                      f'wall_seconds={round(time.time() - chunk_start, 2)}')
            except Exception as exc:
                traceback.print_exc()
                chunk_spec.status = 'failed'
                chunk_failures.append(
                    f'Chunk {chunk_spec.chunk_id} failed: {exc}')
                chunk_results.append(
                    _chunk_result_from_spec(
                        chunk_spec, status='failed', error=str(exc)))
                completed_required_chunks += 1
                print(f'[ERROR] {stats_kind} stats chunk failed: '
                      f'{chunk_spec.chunk_id} '
                      f'progress={completed_required_chunks}/'
                      f'{len(required_chunks)} '
                      f'wall_seconds={round(time.time() - chunk_start, 2)} '
                      f'error={exc}')
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_stats_chunk_worker,
                    stats_kind,
                    processor_config,
                    dataset_files,
                    leads,
                    info_dir,
                    chunk_spec,
                    runtime_settings.pipeline_log_level,
                ): chunk_spec
                for chunk_spec in required_chunks
            }
            future_start_times = {
                future: time.time() for future in futures
            }
            print(f'[INFO] {stats_kind} stats submitted '
                  f'{len(futures)} chunk future(s)')

            for future in as_completed(futures):
                chunk_spec = futures[future]
                try:
                    result = future.result()
                    chunk_spec.status = result['status']
                    chunk_results.append(result)
                    completed_required_chunks += 1
                    print(f'[INFO] {stats_kind} stats chunk completed: '
                          f'{chunk_spec.chunk_id} '
                          f'progress={completed_required_chunks}/'
                          f'{len(required_chunks)} '
                          f'dates={result.get("processed_dates")} '
                          f'max_memory_mb={result.get("max_memory_mb")} '
                          f'elapsed_since_submit_seconds='
                          f'{round(time.time() - future_start_times[future], 2)}')
                except Exception as exc:
                    traceback.print_exc()
                    chunk_spec.status = 'failed'
                    chunk_failures.append(
                        f'Chunk {chunk_spec.chunk_id} failed: {exc}')
                    chunk_results.append(
                        _chunk_result_from_spec(
                            chunk_spec, status='failed', error=str(exc)))
                    completed_required_chunks += 1
                    print(f'[ERROR] {stats_kind} stats chunk failed: '
                          f'{chunk_spec.chunk_id} '
                          f'progress={completed_required_chunks}/'
                          f'{len(required_chunks)} '
                          f'elapsed_since_submit_seconds='
                          f'{round(time.time() - future_start_times[future], 2)} '
                          f'error={exc}')

    chunk_results = sorted(
        chunk_results, key=lambda item: item['chunk_index'])
    write_chunk_plan(plan_path, chunk_specs)

    branch_memory_values = [
        result.get('max_memory_mb') for result in chunk_results
        if result.get('max_memory_mb') is not None
    ]
    branch_max_memory = (
        max(branch_memory_values) if branch_memory_values else
        get_current_memory_mb())
    completed_chunk_count = sum(
        1 for result in chunk_results if result.get('status') == 'success')
    failed_chunk_count = sum(
        1 for result in chunk_results if result.get('status') == 'failed')

    if chunk_failures:
        return {
            'status': 'FAILURE',
            'error': '; '.join(chunk_failures),
            'branch': stats_kind,
            'stats_type': stats_type,
            'chunk_results': chunk_results,
            'chunk_count': len(chunk_specs),
            'required_chunk_count': len(required_chunks),
            'skipped_chunk_count': len(skipped_chunks),
            'completed_chunk_count': completed_chunk_count,
            'failed_chunk_count': failed_chunk_count,
            'max_workers': max_workers,
            'configured_max_workers': configured_max_workers,
            'chunk_size': runtime_settings.pipeline_chunk_size_stats,
            'max_memory_mb': branch_max_memory,
            'output_path': expected_output_path,
        }

    if not StatisticsProcessor.merge_statistics_files(
            stats_type, info_dir, chunk_specs=chunk_specs):
        return {
            'status': 'FAILURE',
            'error': 'statistics chunk merge failed',
            'branch': stats_kind,
            'stats_type': stats_type,
            'chunk_results': chunk_results,
            'chunk_count': len(chunk_specs),
            'required_chunk_count': len(required_chunks),
            'skipped_chunk_count': len(skipped_chunks),
            'completed_chunk_count': completed_chunk_count,
            'failed_chunk_count': failed_chunk_count,
            'max_workers': max_workers,
            'configured_max_workers': configured_max_workers,
            'chunk_size': runtime_settings.pipeline_chunk_size_stats,
            'max_memory_mb': branch_max_memory,
            'output_path': expected_output_path,
        }

    return {
        'status': 'SUCCESS',
        'error': '',
        'branch': stats_kind,
        'stats_type': stats_type,
        'chunk_results': chunk_results,
        'chunk_count': len(chunk_specs),
        'required_chunk_count': len(required_chunks),
        'skipped_chunk_count': len(skipped_chunks),
        'completed_chunk_count': completed_chunk_count,
        'failed_chunk_count': failed_chunk_count,
        'max_workers': max_workers,
        'configured_max_workers': configured_max_workers,
        'chunk_size': runtime_settings.pipeline_chunk_size_stats,
        'max_memory_mb': branch_max_memory,
        'output_path': expected_output_path,
    }
