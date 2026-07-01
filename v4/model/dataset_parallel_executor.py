'''Dataset parallel execution helpers for v4 pipeline mode.'''

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from model.dataset_processor import BatchDatasetProcessor


def _build_chunk_ranges(total_dates, chunk_size):
    '''Build inclusive start/end index chunk ranges.'''
    ranges = []
    for start_idx in range(0, total_dates, chunk_size):
        end_idx = min(total_dates - 1, start_idx + chunk_size - 1)
        ranges.append((start_idx, end_idx))
    return ranges


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


def _run_fcst_chunk_worker(config_path, info_dir, start_idx, end_idx,
                           single_fcst_mode):
    '''Worker entrypoint for one forecast chunk.'''
    processor = BatchDatasetProcessor.from_yaml(config_path, single_fcst_mode)
    processor.ana_model = ''
    processor.clim_model = ''

    results = processor.process_batch(
        target_coll=None,
        info_dir=info_dir,
        date_start_idx=start_idx,
        date_end_idx=end_idx,
        check_only=False,
        skip_calc_mode=False,
        single_fcst_mode=single_fcst_mode,
    )

    if results.get('status') != 'success':
        reason = results.get('reason', 'unknown')
        raise RuntimeError(
            f'Forecast chunk {start_idx}-{end_idx} failed: {reason}')

    if results.get('reason') == 'no_dates_in_chunk':
        return {
            'status': 'skipped',
            'start_idx': start_idx,
            'end_idx': end_idx,
            'processed_dates': 0,
        }

    processor.save_processed_datasets(
        results,
        info_dir=info_dir,
        date_start_idx=start_idx,
        date_end_idx=end_idx,
        target_coll=None,
        single_fcst_mode=bool(single_fcst_mode),
        skip_calc_mode=False,
    )

    return {
        'status': 'success',
        'start_idx': start_idx,
        'end_idx': end_idx,
        'processed_dates': len(results.get('init_dates', [])),
    }


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


def run_parallel_source_dataset_build(config_path, info_dir, runtime_settings,
                                      single_fcst_mode=None):
    '''Run source dataset build with parallel forecast chunk processing.'''
    base_processor = BatchDatasetProcessor.from_yaml(config_path,
                                                     single_fcst_mode)
    process_fcst = bool(base_processor.fcst_model and
                        base_processor.fcst_model.strip())

    chunk_results = []
    built_fcst_this_run = False
    existing_fcst_path = None
    if process_fcst and _forecast_needs_processing(config_path, info_dir,
                                                   single_fcst_mode):
        spacing = base_processor.config.get('fcst_spacing', 1)
        init_dates_full = base_processor._parse_date_range_with_spacing(
            base_processor.config['FDATES'], spacing, [])
        total_dates = len(init_dates_full)
        if total_dates == 0:
            return {
                'status': 'failed',
                'reason': 'no_dates',
                'chunk_results': chunk_results,
            }

        chunk_ranges = _build_chunk_ranges(
            total_dates, runtime_settings.pipeline_chunk_size_fcst)
        max_workers = _resolve_dataset_workers(runtime_settings,
                                               len(chunk_ranges))

        print(f'[INFO] Forecast chunking plan: {len(chunk_ranges)} chunk(s), '
              f'chunk_size={runtime_settings.pipeline_chunk_size_fcst}, '
              f'max_workers={max_workers}')

        chunk_failures = []
        if max_workers == 1:
            for start_idx, end_idx in chunk_ranges:
                try:
                    result = _run_fcst_chunk_worker(
                        config_path, info_dir, start_idx, end_idx,
                        single_fcst_mode)
                    chunk_results.append(result)
                except Exception as exc:
                    traceback.print_exc()
                    chunk_failures.append(
                        f'Chunk {start_idx}-{end_idx} failed: {exc}')
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _run_fcst_chunk_worker,
                        config_path,
                        info_dir,
                        start_idx,
                        end_idx,
                        single_fcst_mode,
                    ): (start_idx, end_idx)
                    for start_idx, end_idx in chunk_ranges
                }

                for future in as_completed(futures):
                    start_idx, end_idx = futures[future]
                    try:
                        chunk_results.append(future.result())
                    except Exception as exc:
                        traceback.print_exc()
                        chunk_failures.append(
                            f'Chunk {start_idx}-{end_idx} failed: {exc}')

        if chunk_failures:
            return {
                'status': 'failed',
                'reason': 'forecast_chunk_failures',
                'errors': chunk_failures,
                'chunk_results': chunk_results,
            }

        if not base_processor.merge_forecast_chunks(
                info_dir, save_for_coll_merge=False):
            return {
                'status': 'failed',
                'reason': 'forecast_chunk_merge_failed',
                'chunk_results': chunk_results,
            }
        built_fcst_this_run = True
    elif process_fcst:
        print('[INFO] Forecast dataset already valid; skipping forecast '
              'chunk processing')
        # Get the existing forecast path
        temp_processor = BatchDatasetProcessor.from_yaml(config_path, single_fcst_mode)
        temp_processor.ana_model = ''
        temp_processor.clim_model = ''
        existing_datasets_check = temp_processor._check_for_existing_datasets()
        existing_fcst_path = existing_datasets_check.get('fcst')

    followup_processor = BatchDatasetProcessor.from_yaml(config_path,
                                                         single_fcst_mode)
    followup_processor.fcst_model = ''

    followup_results = followup_processor.process_batch(
        target_coll=None,
        info_dir=info_dir,
        date_start_idx=None,
        date_end_idx=None,
        check_only=False,
        skip_calc_mode=False,
        single_fcst_mode=single_fcst_mode,
    )

    if followup_results.get('status') != 'success':
        followup_results['chunk_results'] = chunk_results
        return followup_results

    followup_processor.save_processed_datasets(
        followup_results,
        info_dir=info_dir,
        date_start_idx=None,
        date_end_idx=None,
        target_coll=None,
        single_fcst_mode=bool(single_fcst_mode),
        skip_calc_mode=False,
    )

    dataset_files = dict(followup_results.get('dataset_files', {}))
    if built_fcst_this_run:
        dataset_files['fcst'] = os.path.join(
            'outputs', base_processor._generate_output_filenm('fcst'))
    elif existing_fcst_path:  # Use the path we captured earlier
        dataset_files['fcst'] = existing_fcst_path

    return {
        'status': 'success',
        'dataset_files': dataset_files,
        'existing_datasets': followup_results.get('existing_datasets', {}),
        'init_dates': followup_results.get('init_dates', []),
        'leads': followup_results.get('leads', []),
        'chunk_results': chunk_results,
    }

