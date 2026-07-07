'''Statistics parallel execution helpers for v4 pipeline mode.'''

import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from model.chunk_plan import InitDateChunkPlanner, write_chunk_plan
from model.statistics_processor import StatisticsProcessor


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


def _run_stats_chunk_worker(stats_kind, processor_config, dataset_files,
                            leads, info_dir, chunk_spec):
    '''Worker entrypoint for one statistics chunk.'''
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
            'chunk_results': sorted(
                chunk_results, key=lambda item: item['chunk_index']),
            'chunk_count': len(chunk_specs),
            'required_chunk_count': 0,
            'skipped_chunk_count': len(skipped_chunks),
            'max_memory_mb': get_current_memory_mb(),
        }

    max_workers = _resolve_stats_workers(runtime_settings,
                                         len(required_chunks),
                                         stats_kind=stats_kind)
    print(f'[INFO] {stats_kind} stats chunking plan: '
          f'{len(chunk_specs)} chunk(s), {len(required_chunks)} required, '
          f'{len(skipped_chunks)} skipped, '
          f'chunk_size={runtime_settings.pipeline_chunk_size_stats}, '
          f'max_workers={max_workers}')

    chunk_failures = []
    if max_workers == 1:
        for chunk_spec in required_chunks:
            try:
                result = _run_stats_chunk_worker(
                    stats_kind, processor_config, dataset_files, leads,
                    info_dir, chunk_spec)
                chunk_spec.status = result['status']
                chunk_results.append(result)
            except Exception as exc:
                traceback.print_exc()
                chunk_spec.status = 'failed'
                chunk_failures.append(
                    f'Chunk {chunk_spec.chunk_id} failed: {exc}')
                chunk_results.append(
                    _chunk_result_from_spec(
                        chunk_spec, status='failed', error=str(exc)))
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
                ): chunk_spec
                for chunk_spec in required_chunks
            }

            for future in as_completed(futures):
                chunk_spec = futures[future]
                try:
                    result = future.result()
                    chunk_spec.status = result['status']
                    chunk_results.append(result)
                except Exception as exc:
                    traceback.print_exc()
                    chunk_spec.status = 'failed'
                    chunk_failures.append(
                        f'Chunk {chunk_spec.chunk_id} failed: {exc}')
                    chunk_results.append(
                        _chunk_result_from_spec(
                            chunk_spec, status='failed', error=str(exc)))

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

    if chunk_failures:
        return {
            'status': 'FAILURE',
            'error': '; '.join(chunk_failures),
            'chunk_results': chunk_results,
            'chunk_count': len(chunk_specs),
            'required_chunk_count': len(required_chunks),
            'skipped_chunk_count': len(skipped_chunks),
            'max_workers': max_workers,
            'max_memory_mb': branch_max_memory,
        }

    if not StatisticsProcessor.merge_statistics_files(
            stats_type, info_dir, chunk_specs=chunk_specs):
        return {
            'status': 'FAILURE',
            'error': 'statistics chunk merge failed',
            'chunk_results': chunk_results,
            'chunk_count': len(chunk_specs),
            'required_chunk_count': len(required_chunks),
            'skipped_chunk_count': len(skipped_chunks),
            'max_workers': max_workers,
            'max_memory_mb': branch_max_memory,
        }

    return {
        'status': 'SUCCESS',
        'error': '',
        'chunk_results': chunk_results,
        'chunk_count': len(chunk_specs),
        'required_chunk_count': len(required_chunks),
        'skipped_chunk_count': len(skipped_chunks),
        'max_workers': max_workers,
        'max_memory_mb': branch_max_memory,
    }
