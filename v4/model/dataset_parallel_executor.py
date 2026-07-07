'''Dataset parallel execution helpers for v4 pipeline mode.'''

import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

from model.chunk_plan import (
    InitDateChunkPlanner,
    SequenceChunkPlanner,
    write_chunk_plan,
)
from model.dataset_processor import BatchDatasetProcessor


def _resolve_dataset_workers(runtime_settings, num_chunks):
    '''Resolve worker count using runtime settings and SLURM CPU context.'''
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        try:
            default_workers = max(1, int(slurm_cpus))
        except ValueError:
            default_workers = os.cpu_count() or 1
    else:
        default_workers = os.cpu_count() or 1

    configured_workers = runtime_settings.pipeline_max_workers_dataset
    max_workers = configured_workers if configured_workers else default_workers
    max_workers = max(1, min(max_workers, num_chunks))
    return max_workers


def _finish_timing(timings, name, start_time):
    '''Append one elapsed-time record.'''
    timings.append({
        'name': name,
        'wall_seconds': round(time.time() - start_time, 2),
    })


def _chunk_log_prefix(dataset_type):
    '''Return a stable uppercase dataset label for chunk logs.'''
    return dataset_type.upper()


def _log_chunk_queued(dataset_type, chunk_spec, position, total_chunks):
    '''Log that a dataset chunk has been queued/submitted.'''
    print(f'[INFO] {_chunk_log_prefix(dataset_type)} chunk queued '
          f'{position}/{total_chunks}: {chunk_spec.chunk_id} '
          f'items={len(chunk_spec.selected_dates)} '
          f'output={os.path.basename(chunk_spec.output_path)}',
          flush=True)


def _log_chunk_completed(dataset_type, result, completed_count,
                         total_chunks, start_time):
    '''Log that a dataset chunk completed.'''
    elapsed = round(time.time() - start_time, 2)
    output_name = os.path.basename(result.get('output_path', ''))
    print(f'[INFO] {_chunk_log_prefix(dataset_type)} chunk completed '
          f'{completed_count}/{total_chunks}: {result["chunk_id"]} '
          f'status={result["status"]} '
          f'processed_dates={result.get("processed_dates")} '
          f'output={output_name} elapsed_seconds={elapsed}',
          flush=True)


def _log_chunk_failed(dataset_type, chunk_spec, completed_count,
                      total_chunks, exc, start_time):
    '''Log that a dataset chunk failed.'''
    elapsed = round(time.time() - start_time, 2)
    print(f'[ERROR] {_chunk_log_prefix(dataset_type)} chunk failed '
          f'{completed_count}/{total_chunks}: {chunk_spec.chunk_id} '
          f'error={exc} elapsed_seconds={elapsed}',
          flush=True)


def _run_fcst_chunk_worker(config_path, info_dir, chunk_spec,
                           single_fcst_mode):
    '''Worker entrypoint for one forecast chunk.'''
    processor = BatchDatasetProcessor.from_yaml(config_path, single_fcst_mode)
    processor.ana_model = ''
    processor.clim_model = ''

    results = processor.process_batch(
        target_coll=None,
        info_dir=info_dir,
        date_start_idx=chunk_spec.start_idx,
        date_end_idx=chunk_spec.end_idx,
        check_only=False,
        skip_calc_mode=False,
        single_fcst_mode=single_fcst_mode,
    )

    if results.get('status') != 'success':
        reason = results.get('reason', 'unknown')
        raise RuntimeError(
            f'Forecast chunk {chunk_spec.chunk_id} failed: {reason}')

    if results.get('reason') == 'no_dates_in_chunk':
        return {
            'status': 'skipped',
            'chunk_index': chunk_spec.chunk_index,
            'chunk_id': chunk_spec.chunk_id,
            'start_idx': chunk_spec.start_idx,
            'end_idx': chunk_spec.end_idx,
            'output_path': chunk_spec.output_path,
            'processed_dates': 0,
        }

    processor.save_processed_datasets(
        results,
        info_dir=info_dir,
        date_start_idx=chunk_spec.start_idx,
        date_end_idx=chunk_spec.end_idx,
        target_coll=None,
        single_fcst_mode=bool(single_fcst_mode),
        skip_calc_mode=False,
    )

    return {
        'status': 'success',
        'chunk_index': chunk_spec.chunk_index,
        'chunk_id': chunk_spec.chunk_id,
        'start_idx': chunk_spec.start_idx,
        'end_idx': chunk_spec.end_idx,
        'output_path': chunk_spec.output_path,
        'processed_dates': len(results.get('init_dates', [])),
    }


def _time_label(value):
    '''Return stable YYYY-MM-DD_HH label for valid-time chunks.'''
    if isinstance(value, datetime):
        return value.strftime('%Y-%m-%d_%H')
    value_str = str(value).strip()
    if '_' in value_str:
        return value_str
    return datetime.strptime(value_str, '%Y-%m-%d %H:%M:%S').strftime(
        '%Y-%m-%d_%H')


def _parse_time_label(value):
    '''Parse valid-time labels stored in chunk specs.'''
    value_str = str(value).strip()
    for fmt in ('%Y-%m-%d_%H', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
        try:
            return datetime.strptime(value_str, fmt)
        except ValueError:
            pass
    raise ValueError(f'Invalid valid-time chunk label: {value}')


def _resolve_all_valid_times(processor):
    '''Resolve the deterministic valid-time sequence for ana/clim chunks.'''
    spacing = processor.config.get('fcst_spacing', 1)
    init_dates_full = processor._parse_date_range_with_spacing(
        processor.config['FDATES'], spacing, [])
    leads = processor._generate_leads(
        processor.config['FDAYS'], processor.config['NFREQ'])
    valid_times = set()
    for init_date in init_dates_full:
        for fhour in leads:
            valid_times.add(init_date + timedelta(hours=fhour))
    return sorted(valid_times)


def _run_time_chunk_worker(config_path, info_dir, chunk_spec,
                           dataset_type, single_fcst_mode):
    '''Worker entrypoint for one analysis or climatology time chunk.'''
    processor = BatchDatasetProcessor.from_yaml(config_path, single_fcst_mode)
    processor.fcst_model = ''
    if dataset_type == 'ana':
        processor.clim_model = ''
    elif dataset_type == 'clim':
        processor.ana_model = ''
    else:
        raise ValueError(f'Invalid time-chunk dataset type: {dataset_type}')

    valid_times = [
        _parse_time_label(value) for value in chunk_spec.selected_dates
    ]
    process_kwargs = {
        'target_coll': None,
        'info_dir': info_dir,
        'date_start_idx': None,
        'date_end_idx': None,
        'check_only': False,
        'skip_calc_mode': False,
        'single_fcst_mode': single_fcst_mode,
    }
    if dataset_type == 'ana':
        process_kwargs['ana_valid_times'] = valid_times
    else:
        process_kwargs['clim_valid_times'] = valid_times

    results = processor.process_batch(**process_kwargs)
    if results.get('status') != 'success':
        reason = results.get('reason', 'unknown')
        raise RuntimeError(
            f'{dataset_type} chunk {chunk_spec.chunk_id} failed: {reason}')

    processor.save_processed_datasets(
        results,
        info_dir=info_dir,
        target_coll=None,
        single_fcst_mode=bool(single_fcst_mode),
        skip_calc_mode=False,
        chunk_id=chunk_spec.chunk_id,
    )

    return {
        'status': 'success',
        'chunk_index': chunk_spec.chunk_index,
        'chunk_id': chunk_spec.chunk_id,
        'start_idx': chunk_spec.start_idx,
        'end_idx': chunk_spec.end_idx,
        'output_path': chunk_spec.output_path,
        'processed_dates': len(valid_times),
        'dataset_type': dataset_type,
    }


def _chunk_result_from_spec(chunk_spec, status=None, error=None):
    '''Build a deterministic result record from a chunk spec.'''
    result = {
        'status': status or chunk_spec.status,
        'chunk_index': chunk_spec.chunk_index,
        'chunk_id': chunk_spec.chunk_id,
        'start_idx': chunk_spec.start_idx,
        'end_idx': chunk_spec.end_idx,
        'output_path': chunk_spec.output_path,
        'processed_dates': len(chunk_spec.selected_dates),
    }
    if error:
        result['error'] = error
    return result


def _resolve_final_dataset_files(processor, results, dataset_types):
    '''Resolve usable final dataset paths for existing and newly saved files.'''
    dataset_files = {}
    existing_datasets = results.get('existing_datasets', {})
    new_datasets = results.get('datasets', {})

    for dataset_type in dataset_types:
        if dataset_type in existing_datasets:
            dataset_files[dataset_type] = existing_datasets[dataset_type]
        elif dataset_type in new_datasets:
            dataset_files[dataset_type] = os.path.join(
                'outputs', processor._generate_output_filenm(dataset_type))

    return dataset_files


def _forecast_needs_processing(config_path, info_dir, single_fcst_mode):
    '''Check whether forecast dataset creation is required.'''
    checker = BatchDatasetProcessor.from_yaml(config_path, single_fcst_mode)
    checker.ana_model = ''
    checker.clim_model = ''
    check_result = checker.process_batch(
        target_coll=None,
        info_dir=info_dir,
        date_start_idx=None,
        date_end_idx=None,
        check_only=True,
        skip_calc_mode=False,
        single_fcst_mode=single_fcst_mode,
    )
    datasets_needed = check_result.get('datasets_needed', [])
    return 'fcst' in datasets_needed


def _dataset_needs_processing(config_path, info_dir, single_fcst_mode,
                              dataset_type):
    '''Check whether one source dataset type needs creation.'''
    checker = BatchDatasetProcessor.from_yaml(config_path, single_fcst_mode)
    checker.fcst_model = ''
    if dataset_type == 'ana':
        checker.clim_model = ''
    elif dataset_type == 'clim':
        checker.ana_model = ''
    else:
        raise ValueError(f'Invalid dataset type: {dataset_type}')

    check_result = checker.process_batch(
        target_coll=None,
        info_dir=info_dir,
        date_start_idx=None,
        date_end_idx=None,
        check_only=True,
        skip_calc_mode=False,
        single_fcst_mode=single_fcst_mode,
    )
    datasets_needed = check_result.get('datasets_needed', [])
    return dataset_type in datasets_needed


def _run_required_chunks(required_chunks, worker_func, worker_args,
                         dataset_type):
    '''Run required chunks sequentially or in a process pool.'''
    worker_args = dict(worker_args)
    max_workers = worker_args.pop('max_workers')
    chunk_results = []
    chunk_failures = []
    total_chunks = len(required_chunks)
    start_time = time.time()
    completed_count = 0

    print(f'[INFO] {_chunk_log_prefix(dataset_type)} chunk execution starting: '
          f'{total_chunks} required chunk(s), max_workers={max_workers}',
          flush=True)

    if max_workers == 1:
        for position, chunk_spec in enumerate(required_chunks, start=1):
            _log_chunk_queued(dataset_type, chunk_spec, position,
                              total_chunks)
            try:
                result = worker_func(chunk_spec=chunk_spec, **worker_args)
                chunk_spec.status = result['status']
                chunk_results.append(result)
                completed_count += 1
                _log_chunk_completed(dataset_type, result, completed_count,
                                     total_chunks, start_time)
            except Exception as exc:
                traceback.print_exc()
                chunk_spec.status = 'failed'
                error = f'Chunk {chunk_spec.chunk_id} failed: {exc}'
                chunk_failures.append(error)
                chunk_results.append(
                    _chunk_result_from_spec(
                        chunk_spec, status='failed', error=str(exc)))
                completed_count += 1
                _log_chunk_failed(dataset_type, chunk_spec, completed_count,
                                  total_chunks, exc, start_time)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for position, chunk_spec in enumerate(required_chunks, start=1):
                _log_chunk_queued(dataset_type, chunk_spec, position,
                                  total_chunks)
                future = executor.submit(
                    worker_func, chunk_spec=chunk_spec, **worker_args)
                futures[future] = chunk_spec
            for future in as_completed(futures):
                chunk_spec = futures[future]
                try:
                    result = future.result()
                    chunk_spec.status = result['status']
                    chunk_results.append(result)
                    completed_count += 1
                    _log_chunk_completed(dataset_type, result,
                                         completed_count, total_chunks,
                                         start_time)
                except Exception as exc:
                    traceback.print_exc()
                    chunk_spec.status = 'failed'
                    error = f'Chunk {chunk_spec.chunk_id} failed: {exc}'
                    chunk_failures.append(error)
                    chunk_results.append(
                        _chunk_result_from_spec(
                            chunk_spec, status='failed', error=str(exc)))
                    completed_count += 1
                    _log_chunk_failed(dataset_type, chunk_spec,
                                      completed_count, total_chunks, exc,
                                      start_time)

    return chunk_results, chunk_failures


def _run_time_dataset_build(config_path, info_dir, runtime_settings,
                            dataset_type, single_fcst_mode=None):
    '''Run analysis or climatology build in deterministic valid-time chunks.'''
    timings = []
    overall_start = time.time()
    base_processor = BatchDatasetProcessor.from_yaml(config_path,
                                                     single_fcst_mode)
    process_dataset = bool(getattr(base_processor, f'{dataset_type}_model')
                           and getattr(base_processor,
                                       f'{dataset_type}_model').strip())
    if not process_dataset:
        _finish_timing(timings, f'{dataset_type}_total', overall_start)
        return {
            'status': 'success',
            'dataset_files': {},
            'existing_datasets': {},
            'chunk_results': [],
            'timings': timings,
        }

    check_start = time.time()
    if not _dataset_needs_processing(config_path, info_dir, single_fcst_mode,
                                     dataset_type):
        _finish_timing(timings, f'{dataset_type}_existing_dataset_check',
                       check_start)
        print(f'[INFO] {dataset_type.upper()} dataset already valid; '
              f'skipping {dataset_type} chunk processing')
        existing_datasets_check = base_processor._check_for_existing_datasets()
        dataset_files = {}
        existing_datasets = {}
        if existing_datasets_check.get(dataset_type):
            dataset_files[dataset_type] = existing_datasets_check[dataset_type]
            existing_datasets[dataset_type] = existing_datasets_check[
                dataset_type]
        _finish_timing(timings, f'{dataset_type}_total', overall_start)
        return {
            'status': 'success',
            'dataset_files': dataset_files,
            'existing_datasets': existing_datasets,
            'chunk_results': [],
            'timings': timings,
        }
    _finish_timing(timings, f'{dataset_type}_existing_dataset_check',
                   check_start)

    valid_times = _resolve_all_valid_times(base_processor)
    if not valid_times:
        _finish_timing(timings, f'{dataset_type}_total', overall_start)
        return {
            'status': 'failed',
            'reason': f'no_{dataset_type}_valid_times',
            'chunk_results': [],
            'timings': timings,
        }

    chunk_output_dir = os.path.join('outputs', str(info_dir), 'tmp')
    chunk_size = (
        runtime_settings.pipeline_chunk_size_ana
        if dataset_type == 'ana'
        else runtime_settings.pipeline_chunk_size_clim
    )
    chunk_specs = SequenceChunkPlanner(
        valid_times, label_func=_time_label).build(
            chunk_size,
            chunk_output_dir,
            dataset_type,
        )
    plan_path = os.path.join(
        chunk_output_dir, f'chunk_plan_{dataset_type}.json')
    write_chunk_plan(plan_path, chunk_specs)
    required_chunks = [
        chunk_spec for chunk_spec in chunk_specs if chunk_spec.is_required
    ]
    skipped_chunks = [
        chunk_spec for chunk_spec in chunk_specs if chunk_spec.is_skipped
    ]
    chunk_results = [
        _chunk_result_from_spec(chunk_spec) for chunk_spec in skipped_chunks
    ]
    if not required_chunks:
        _finish_timing(timings, f'{dataset_type}_total', overall_start)
        return {
            'status': 'failed',
            'reason': f'no_{dataset_type}_chunks',
            'chunk_results': chunk_results,
            'timings': timings,
        }

    max_workers = _resolve_dataset_workers(runtime_settings,
                                           len(required_chunks))
    print(f'[INFO] {dataset_type.upper()} chunking plan: '
          f'{len(chunk_specs)} chunk(s), {len(required_chunks)} required, '
          f'{len(skipped_chunks)} skipped, chunk_size={chunk_size}, '
          f'max_workers={max_workers}')

    chunk_start = time.time()
    new_results, chunk_failures = _run_required_chunks(
        required_chunks,
        _run_time_chunk_worker,
        {
            'config_path': config_path,
            'info_dir': info_dir,
            'dataset_type': dataset_type,
            'single_fcst_mode': single_fcst_mode,
            'max_workers': max_workers,
        },
        dataset_type,
    )
    _finish_timing(timings, f'{dataset_type}_chunk_execution', chunk_start)
    chunk_results.extend(new_results)
    chunk_results = sorted(chunk_results, key=lambda item: item['chunk_index'])
    write_chunk_plan(plan_path, chunk_specs)

    if chunk_failures:
        _finish_timing(timings, f'{dataset_type}_total', overall_start)
        return {
            'status': 'failed',
            'reason': f'{dataset_type}_chunk_failures',
            'errors': chunk_failures,
            'chunk_results': chunk_results,
            'timings': timings,
        }

    merge_start = time.time()
    if not base_processor.merge_time_chunks(dataset_type, info_dir,
                                            chunk_specs):
        _finish_timing(timings, f'{dataset_type}_chunk_merge', merge_start)
        _finish_timing(timings, f'{dataset_type}_total', overall_start)
        return {
            'status': 'failed',
            'reason': f'{dataset_type}_chunk_merge_failed',
            'chunk_results': chunk_results,
            'timings': timings,
        }
    _finish_timing(timings, f'{dataset_type}_chunk_merge', merge_start)
    _finish_timing(timings, f'{dataset_type}_total', overall_start)

    return {
        'status': 'success',
        'dataset_files': {
            dataset_type: os.path.join(
                'outputs',
                base_processor._generate_output_filenm(dataset_type))
        },
        'existing_datasets': {},
        'chunk_results': chunk_results,
        'timings': timings,
    }


def run_parallel_source_dataset_build(config_path, info_dir, runtime_settings,
                                      single_fcst_mode=None):
    '''Run source dataset build with parallel forecast chunk processing.'''
    timings = []
    source_start = time.time()
    base_processor = BatchDatasetProcessor.from_yaml(config_path,
                                                     single_fcst_mode)
    process_fcst = bool(base_processor.fcst_model and
                        base_processor.fcst_model.strip())

    chunk_results = []
    forecast_results = {
        'existing_datasets': {},
        'datasets': {},
    }
    if process_fcst and _forecast_needs_processing(config_path, info_dir,
                                                   single_fcst_mode):
        spacing = base_processor.config.get('fcst_spacing', 1)
        init_dates_full = base_processor._parse_date_range_with_spacing(
            base_processor.config['FDATES'], spacing, [])
        total_dates = len(init_dates_full)
        if total_dates == 0:
            _finish_timing(timings, 'source_dataset_build_total',
                           source_start)
            return {
                'status': 'failed',
                'reason': 'no_dates',
                'chunk_results': chunk_results,
                'timings': timings,
            }

        chunk_output_dir = os.path.join('outputs', str(info_dir), 'tmp')
        chunk_specs = InitDateChunkPlanner(
            init_dates_full,
            exclude_dates=base_processor.config.get('exclude_dates', []),
        ).build(
            runtime_settings.pipeline_chunk_size_fcst,
            chunk_output_dir,
            'fcst',
        )
        plan_path = os.path.join(chunk_output_dir, 'chunk_plan_fcst.json')
        write_chunk_plan(plan_path, chunk_specs)

        required_chunks = [
            chunk_spec for chunk_spec in chunk_specs
            if chunk_spec.is_required
        ]
        skipped_chunks = [
            chunk_spec for chunk_spec in chunk_specs
            if chunk_spec.is_skipped
        ]
        if skipped_chunks:
            chunk_results.extend(
                _chunk_result_from_spec(chunk_spec)
                for chunk_spec in skipped_chunks
            )
        if not required_chunks:
            write_chunk_plan(plan_path, chunk_specs)
            _finish_timing(timings, 'source_dataset_build_total',
                           source_start)
            return {
                'status': 'failed',
                'reason': 'no_forecast_dates_after_exclusions',
                'chunk_results': sorted(
                    chunk_results, key=lambda item: item['chunk_index']),
                'timings': timings,
            }

        max_workers = _resolve_dataset_workers(runtime_settings,
                                               len(required_chunks))

        print(f'[INFO] Forecast chunking plan: {len(chunk_specs)} chunk(s), '
              f'{len(required_chunks)} required, '
              f'{len(skipped_chunks)} skipped, '
              f'chunk_size={runtime_settings.pipeline_chunk_size_fcst}, '
              f'max_workers={max_workers}')

        chunk_start = time.time()
        new_results, chunk_failures = _run_required_chunks(
            required_chunks,
            _run_fcst_chunk_worker,
            {
                'config_path': config_path,
                'info_dir': info_dir,
                'single_fcst_mode': single_fcst_mode,
                'max_workers': max_workers,
            },
            'fcst',
        )
        chunk_results.extend(new_results)
        _finish_timing(timings, 'fcst_chunk_execution', chunk_start)

        chunk_results = sorted(
            chunk_results, key=lambda item: item['chunk_index'])
        write_chunk_plan(plan_path, chunk_specs)

        if chunk_failures:
            _finish_timing(timings, 'source_dataset_build_total',
                           source_start)
            return {
                'status': 'failed',
                'reason': 'forecast_chunk_failures',
                'errors': chunk_failures,
                'chunk_results': chunk_results,
                'timings': timings,
            }

        merge_start = time.time()
        if not base_processor.merge_forecast_chunks(
                info_dir, save_for_coll_merge=False,
                chunk_specs=chunk_specs):
            _finish_timing(timings, 'fcst_chunk_merge', merge_start)
            _finish_timing(timings, 'source_dataset_build_total',
                           source_start)
            return {
                'status': 'failed',
                'reason': 'forecast_chunk_merge_failed',
                'chunk_results': chunk_results,
                'timings': timings,
            }
        _finish_timing(timings, 'fcst_chunk_merge', merge_start)
        forecast_results['datasets']['fcst'] = True
    elif process_fcst:
        check_start = time.time()
        print('[INFO] Forecast dataset already valid; skipping forecast '
              'chunk processing')
        temp_processor = BatchDatasetProcessor.from_yaml(config_path,
                                                         single_fcst_mode)
        temp_processor.ana_model = ''
        temp_processor.clim_model = ''
        existing_datasets_check = temp_processor._check_for_existing_datasets()
        if existing_datasets_check.get('fcst'):
            forecast_results['existing_datasets']['fcst'] = (
                existing_datasets_check['fcst'])
        _finish_timing(timings, 'fcst_existing_dataset_check', check_start)

    ana_results = _run_time_dataset_build(
        config_path, info_dir, runtime_settings, 'ana',
        single_fcst_mode=single_fcst_mode)
    timings.extend(ana_results.get('timings', []))
    chunk_results.extend(ana_results.get('chunk_results', []))
    if ana_results.get('status') != 'success':
        ana_results['chunk_results'] = sorted(
            chunk_results, key=lambda item: item['chunk_index'])
        _finish_timing(timings, 'source_dataset_build_total', source_start)
        ana_results['timings'] = timings
        return ana_results

    clim_results = _run_time_dataset_build(
        config_path, info_dir, runtime_settings, 'clim',
        single_fcst_mode=single_fcst_mode)
    timings.extend(clim_results.get('timings', []))
    chunk_results.extend(clim_results.get('chunk_results', []))
    if clim_results.get('status') != 'success':
        clim_results['chunk_results'] = sorted(
            chunk_results, key=lambda item: item['chunk_index'])
        _finish_timing(timings, 'source_dataset_build_total', source_start)
        clim_results['timings'] = timings
        return clim_results

    dataset_files = {}
    dataset_files.update(_resolve_final_dataset_files(
        base_processor, forecast_results, ['fcst']))
    dataset_files.update(ana_results.get('dataset_files', {}))
    dataset_files.update(clim_results.get('dataset_files', {}))

    final_processor = BatchDatasetProcessor.from_yaml(config_path,
                                                      single_fcst_mode)
    spacing = final_processor.config.get('fcst_spacing', 1)
    exclude_dates = final_processor.config.get('exclude_dates', [])
    init_dates = final_processor._parse_date_range_with_spacing(
        final_processor.config['FDATES'], spacing, exclude_dates)
    leads = final_processor._generate_leads(
        final_processor.config['FDAYS'], final_processor.config['NFREQ'])
    _finish_timing(timings, 'source_dataset_build_total', source_start)

    return {
        'status': 'success',
        'dataset_files': dataset_files,
        'existing_datasets': {
            **ana_results.get('existing_datasets', {}),
            **clim_results.get('existing_datasets', {}),
        },
        'init_dates': init_dates,
        'leads': leads,
        'chunk_results': chunk_results,
        'timings': timings,
    }

