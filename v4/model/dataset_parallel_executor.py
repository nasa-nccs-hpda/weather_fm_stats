'''Dataset parallel execution helpers for v4 pipeline mode.'''

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from model.chunk_plan import InitDateChunkPlanner, write_chunk_plan
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
            return {
                'status': 'failed',
                'reason': 'no_forecast_dates_after_exclusions',
                'chunk_results': sorted(
                    chunk_results, key=lambda item: item['chunk_index']),
            }

        max_workers = _resolve_dataset_workers(runtime_settings,
                                               len(required_chunks))

        print(f'[INFO] Forecast chunking plan: {len(chunk_specs)} chunk(s), '
              f'{len(required_chunks)} required, '
              f'{len(skipped_chunks)} skipped, '
              f'chunk_size={runtime_settings.pipeline_chunk_size_fcst}, '
              f'max_workers={max_workers}')

        chunk_failures = []
        if max_workers == 1:
            for chunk_spec in required_chunks:
                try:
                    result = _run_fcst_chunk_worker(
                        config_path, info_dir, chunk_spec,
                        single_fcst_mode)
                    chunk_spec.status = result['status']
                    chunk_results.append(result)
                except Exception as exc:
                    traceback.print_exc()
                    chunk_spec.status = 'failed'
                    error = f'Chunk {chunk_spec.chunk_id} failed: {exc}'
                    chunk_failures.append(error)
                    chunk_results.append(
                        _chunk_result_from_spec(
                            chunk_spec, status='failed', error=str(exc)))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _run_fcst_chunk_worker,
                        config_path,
                        info_dir,
                        chunk_spec,
                        single_fcst_mode,
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
                        error = f'Chunk {chunk_spec.chunk_id} failed: {exc}'
                        chunk_failures.append(error)
                        chunk_results.append(
                            _chunk_result_from_spec(
                                chunk_spec, status='failed', error=str(exc)))

        chunk_results = sorted(
            chunk_results, key=lambda item: item['chunk_index'])
        write_chunk_plan(plan_path, chunk_specs)

        if chunk_failures:
            return {
                'status': 'failed',
                'reason': 'forecast_chunk_failures',
                'errors': chunk_failures,
                'chunk_results': chunk_results,
            }

        if not base_processor.merge_forecast_chunks(
                info_dir, save_for_coll_merge=False,
                chunk_specs=chunk_specs):
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

