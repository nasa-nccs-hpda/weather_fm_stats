'''CLI controller for selecting and running stats workflows.'''

import argparse
import os
import sys
import traceback
from datetime import datetime

import xarray as xr

from model.constants import VARS_LONG_MAP, VARS_SCALE_MAP, VARS_UNIT_MAP
from model.dataset_processor import BatchDatasetProcessor
from model.statistics_processor import StatisticsProcessor

# ================== MAIN FUNCTION ==================


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
    if not (args.fcst or args.ana or args.clim or args.process or args.stats or
        args.merge_collections or args.merge_forecast_chunks or args.clean):
        args.process = True


def validate_runtime_args(args, single_fcst_mode):
    '''Validate regular processing, statistics, and merge argument combinations.'''
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


def main():
    '''Main function'''

    # Parse arguments
    args = parse_arguments()
    single_fcst_mode = args.date

    using_chunks = prepare_runtime(args, single_fcst_mode)
    skip_calc_mode = bool(args.target_coll)

    try:
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

            # Save new datasets if they were created
            if single_fcst_mode:
                # Apply scaling for in-memory datasets
                datasets = results['datasets']
                for ds_nm, ds in datasets.items():
                    for var in ds.data_vars:
                        if var != 'grid_weights':
                            var_upper = var.upper()
                            if var_upper in VARS_SCALE_MAP:
                                scale_factor = VARS_SCALE_MAP[var_upper]
                                if scale_factor != 1.0:
                                    ds[var] = ds[var] * scale_factor
                print('\n[INFO] Single-forecast mode: Keeping datasets in '
                      'memory (not saving to disk)')
            elif 'datasets' in results:
                print('\n--- Saving datasets ---')
                datasets = results['datasets']
                existing_datasets = results.get('existing_datasets', {})
                saved_count = 0
                for ds_nm, ds in datasets.items():
                    if ds_nm not in existing_datasets:
                        # Chunk filename (save in tmp)
                        if (args.date_start_idx is not None or
                            args.date_end_idx is not None):
                            chunk_id = (f'chunk_{args.date_start_idx:03d}_'
                                        f'{args.date_end_idx:03d}')
                            filenm = os.path.join(
                                f'outputs/{args.info_dir}/tmp',
                                processor._generate_output_filenm(
                                    ds_nm, chunk_id=chunk_id,
                                    info_dir=args.info_dir))
                        # Collection-specific filename (save in tmp)
                        elif args.target_coll:
                            filenm = os.path.join(
                                f'outputs/{args.info_dir}/tmp',
                                processor._generate_output_filenm(
                                    ds_nm, target_coll=args.target_coll,
                                    info_dir=args.info_dir))
                        # Regular filename
                        else:
                            filenm = os.path.join(
                                'outputs',
                                processor._generate_output_filenm(ds_nm))

                        # Define coordinates and dimensions to keep
                        if ds_nm == 'fcst':
                            cds_to_keep = ['init_date', 'lead', 'lev', 'lat',
                                           'lon']
                        elif ds_nm == 'ana':
                            cds_to_keep = ['time', 'lev', 'lat', 'lon']
                        elif ds_nm == 'clim':
                            cds_to_keep = ['time', 'lev', 'lat', 'lon']

                        # Create clean dataset with only required dims/coords
                        data_vars = {}
                        for var in ds.data_vars:
                            if var != 'grid_weights':
                                var_data = ds[var]
                                dims_to_drop = [dim for dim in var_data.dims
                                                if dim not in cds_to_keep]
                                for dim in dims_to_drop:
                                    if dim in var_data.dims:
                                        var_data = var_data.isel({dim: 0},
                                                                 drop=True)
                                var_data = var_data.astype('float32')
                                # Apply scaling and metadata
                                # Skip in collection-specific mode
                                if not skip_calc_mode:
                                    var_upper = var.upper()
                                    if var_upper in VARS_SCALE_MAP:
                                        scale_factor = (
                                            VARS_SCALE_MAP[var_upper])
                                        if scale_factor != 1.0:
                                            var_data = var_data * scale_factor
                                    if var_upper in VARS_LONG_MAP:
                                        var_data.attrs['long_name'] = (
                                            VARS_LONG_MAP[var_upper])
                                    if var_upper in VARS_UNIT_MAP:
                                        var_data.attrs['units'] = (
                                            VARS_UNIT_MAP[var_upper])
                                data_vars[var] = var_data
                        clean_coords = {coord: ds.coords[coord] for coord
                                        in cds_to_keep if coord in ds.coords}
                        ds_clean = xr.Dataset(data_vars, coords=clean_coords)
                        unwanted_dims = [dim for dim in ds_clean.dims
                                         if dim not in cds_to_keep]
                        if unwanted_dims:
                            ds_clean = ds_clean.drop_dims(unwanted_dims)

                        # Add grid weights back and save with compression
                        ds_clean['grid_weights'] = ds['grid_weights']
                        encoding = {}
                        for var in ds_clean.variables:
                            encoding[var] = {'zlib': True, 'complevel': 2}
                            dtype_nm = str(ds_clean[var].dtype)
                            if 'float64' in dtype_nm:
                                encoding[var]['dtype'] = 'float32'
                        # Add exclusion metadata for forecast datasets
                        if ds_nm == 'fcst':
                            BatchDatasetProcessor._add_exclusion_metadata(
                                ds_clean, processor.config)

                        # Make 'init_date' unlimited if fcst chunk
                        if (args.date_start_idx is not None and
                            ds_nm == 'fcst'):
                            ds_clean.to_netcdf(
                                filenm, unlimited_dims=['init_date'],
                                encoding=encoding)
                        else:
                            ds_clean.to_netcdf(filenm, encoding=encoding)

                # Print summary
                        final_dims = dict(ds_clean.sizes)
                        print(f'  [OK] Saved: {filenm}')
                        print(f'    Variables: {list(ds.data_vars.keys())}')
                        print(f'    Dimensions: {final_dims}')
                        saved_count += 1
                    else:
                        print(f'  [INFO] Using existing: '
                              f'{os.path.basename(existing_datasets[ds_nm])}')
                if saved_count > 0:
                    print(f'\n[SUCCESS] Saved {saved_count} new datasets!')
                if existing_datasets:
                    print(f'[INFO] Used {len(existing_datasets)} existing '
                          f'datasets')

        # Phase 2: Statistics Calculation
        if args.stats or args.date: # also run in single-fcst mode
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
