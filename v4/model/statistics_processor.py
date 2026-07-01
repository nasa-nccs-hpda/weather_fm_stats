'''Statistics processing model.'''

import gc
import glob
import os
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from model.constants import *
from model.dataset_processor import BatchDatasetProcessor

# ================== STATISTICS PROCESSOR CLASS ==================


class StatisticsProcessor:
    '''Statistics calculation processor for regional and global stats'''

    def __init__(self, config: Dict[str, Any], dataset_files = None,
                 in_memory_datasets = None, init_dates: List[datetime] = None,
                 leads: List[int] = None):
        '''Initialize statistics processor with required data'''
        self.config = config
        self.dataset_files = dataset_files
        self.datasets = in_memory_datasets
        self.init_dates = init_dates
        self.leads = leads
        self.enhanced_weights = None
        self.stats_arrays = None
        self.regional_masks = None
        self.required_vars = self._get_required_vars_from_config()
        self.required_levels = config['levels']

    def _get_required_vars_from_config(self) -> List[str]:
        '''Extract required variables from config'''
        required_vars = []
        for key, values in self.config.items():
            if (key.startswith(('2d_vars_', '3d_vars_'))
                and isinstance(values, list)):
                for var in values:
                    if isinstance(var, str):
                        var_nms = var.split('#')[0].strip()
                        if var_nms:
                            required_vars.append(var_nms)
                    else:
                        required_vars.append(var)
        return list(set(required_vars))

    def _load_and_validate_datasets(self):
        '''Load datasets and validate required variables/levels'''

        # Check if datasets are already provided (single-forecast mode)
        if self.datasets is not None:
            print('Using in-memory datasets for statistics...')
            # Skip the file loading section entirely
        else:
            # Load datasets from files (regular mode)
            print('Loading datasets for statistics...')
            self.datasets = {}
            for dataset_type, file_path in self.dataset_files.items():
                if not os.path.exists(file_path):
                    raise ValueError(
                        f'{dataset_type} file not found: {file_path}')
                print(f'  Loading {dataset_type}: {file_path}')
                self.datasets[dataset_type] = xr.open_dataset(
                    file_path, decode_timedelta=True)
                print(f'    Variables: '
                    f'{list(self.datasets[dataset_type].data_vars.keys())}')
                print(f'    Dimensions: '
                    f'{dict(self.datasets[dataset_type].sizes)}')

        # Validate required variables and levels
        print('  Validating datasets for required variables and levels...')
        for dataset_type, ds in self.datasets.items():
            missing_vars = [var for var in self.required_vars
                            if var not in ds.data_vars]
            if missing_vars:
                raise ValueError(f'{dataset_type} dataset missing required '
                                f'variables: {missing_vars}')

            if 'lev' in ds.coords:
                missing_levels = [lev for lev in self.required_levels
                                if lev not in ds.lev.values]
                if missing_levels:
                    raise ValueError(f'{dataset_type} dataset missing '
                                f'required levels: {missing_levels}')
        print('  [OK] All datasets contain required variables and levels')

        # Filter datasets to only include requested pressure levels
        datasets_filtered = False
        for dataset_type, ds in list(self.datasets.items()):
            if 'lev' in ds.coords and len(ds.lev) > len(self.required_levels):
                if not datasets_filtered:
                    print('  Filtering datasets to requested levels...')
                    datasets_filtered = True
                print(f'    {dataset_type}: Filtering from {len(ds.lev)} '
                    f'to {len(self.required_levels)} requested levels')
                # Filter dataset to only include requested pressure levels
                filtered_ds = ds.sel(lev=self.required_levels)
                self.datasets[dataset_type] = filtered_ds
        if datasets_filtered:
            print('  [OK] All datasets filtered to requested pressure levels')

    def _create_enhanced_weights(self):
        '''Create enhanced grid weights with NaN masking'''
        print('Creating enhanced grid weights with NaN masking...')

        # Get original grid weights from fcst (representative of any dataset)
        base_weights = self.datasets['fcst']['grid_weights']

        # Find sample 3d and 2d variables
        sample_3d_var = None
        sample_2d_var = None
        forecast_ds = self.datasets['fcst']
        for var_nms in forecast_ds.data_vars:
            if var_nms == 'grid_weights':
                continue
            var = forecast_ds[var_nms]
            if 'lev' in var.dims and sample_3d_var is None:
                sample_3d_var = var_nms
            elif 'lev' not in var.dims and sample_2d_var is None:
                sample_2d_var = var_nms
            if sample_3d_var and sample_2d_var:
                break
        print(f'  Using sample variables: 3D={sample_3d_var}, '
              f'2D={sample_2d_var}')

        # Process each dataset
        nan_mask_3d = None
        nan_mask_2d = None
        for ds_nm, dataset in self.datasets.items():
            print(f'  Checking NaN patterns in {ds_nm} dataset...')
            for var_name, current_mask in [(sample_3d_var, nan_mask_3d),
                                           (sample_2d_var, nan_mask_2d)]:
                if var_name and var_name in dataset.data_vars:
                    var_data = dataset[var_name]
                    dims = ['init_date', 'lead'
                            ] if ds_nm == 'fcst' else ['time']
                    is_nan = var_data.isnull().any(dim=dims)
                    if var_name == sample_3d_var:
                        nan_mask_3d = is_nan if nan_mask_3d is None else (
                            nan_mask_3d | is_nan)
                    else:
                        nan_mask_2d = is_nan if nan_mask_2d is None else (
                            nan_mask_2d | is_nan)

        # Create enhanced weights
        self.enhanced_weights = {}
        for mask, weight_type, dims_info in [
            (nan_mask_3d, 'weights_3d', '3D'),
            (nan_mask_2d, 'weights_2d', '2D')
        ]:
            if mask is not None:
                if weight_type == 'weights_3d':
                    weights = base_weights.broadcast_like(mask).where(~mask)
                else:
                    weights = base_weights.where(~mask)
                weights.name = f'enhanced_grid_weights_{dims_info.lower()}'
                self.enhanced_weights[weight_type] = weights
                print(f'    Created {dims_info} weights: '
                      f'{dict(weights.sizes)}')

    def run_regional_statistics(self, info_dir=None, using_chunks=False,
                                chunk_number=None, skip_avg=False,
                                save_dir=None):
        '''Run complete regional statistics workflow'''
        print('\n=== Regional Statistics Processing ===')

        # Load/validate datasets, create regional masks, and initialize arrays
        self._load_and_validate_datasets()
        requested_regions = self.config['regions']
        self.regional_masks = self._create_regional_masks(requested_regions)
        print(f'Created {len(self.regional_masks)} regional masks')
        #self.save_masks_for_testing(self.regional_masks) # for DEBUG
        self.stats_arrays = self._initialize_regional_stats_arrays()
        print(f'Initialized statistics arrays for: '
              f'{STATS_TO_CALCULATE_REGIONAL}')

        # Calculate statistics with enhanced weights
        self._create_enhanced_weights()
        self._calculate_statistics_loop('reg')
        if not skip_avg:
            self._compute_averaged_statistics()

        # Set save_dir for single-forecast mode
        if save_dir:
            self.save_dir = save_dir

        # Save results and close datasets
        self._save_statistics_output('reg', info_dir, using_chunks,
                                     chunk_number, skip_avg)
        for ds in self.datasets.values():
            ds.close()
        print('[SUCCESS] Regional statistics analysis completed!')

    def run_global_statistics(self, info_dir=None, using_chunks=False,
                              chunk_number=None, skip_avg=False):
        '''Run complete global statistics workflow'''
        print('\n=== Global Statistics Processing ===')

        # Load/validate datasets and initialize statistics arrays
        self._load_and_validate_datasets()
        self.stats_arrays = self._initialize_global_stats_arrays()
        print(f'Initialized statistics arrays for: '
              f'{STATS_TO_CALCULATE_GLOBAL}')

        # Calculate statistics with enhanced weights
        self._create_enhanced_weights()
        self._calculate_statistics_loop('glo')
        if not skip_avg:
            self._compute_averaged_statistics()
            self._compute_global_avg_statistic()

        # Save results and close datasets
        self._save_statistics_output('glo', info_dir, using_chunks,
                                     chunk_number, skip_avg)
        for ds in self.datasets.values():
            ds.close()
        print('[SUCCESS] Global statistics analysis completed!')

    def _create_regional_masks(self, requested_regions):
        '''Create regional masks based on lat/lon bounds or NetCDF files'''

        # Region definitions for coordinate-based masks (N/S/E/W bounds)
        local_region_coords_map = {
            'GLO': [90.0, -90.0, 180.0, -180.0],
            'NHE': [80.0, 20.0, 180.0, -180.0],
            'TRO': [20.0, -20.0, 180.0, -180.0],
            'SHE': [-20.0, -80.0, 180.0, -180.0],
            'NWQ': [90.0, 0.0, 0.0, -180.0],
            'NEQ': [90.0, 0.0, 180.0, 0.0],
            'SWQ': [0.0, -90.0, 0.0, -180.0],
            'SEQ': [0.0, -90.0, 180.0, 0.0],
            'NAM': [60.0, 20.0, -60.0, -140.0],
            'EUR': [60.0, 30.0, 30.0, -10.0],
            'NPO': [90.0, 60.0, 180.0, -180.0],
            'SPO': [-60.0, -90.0, 180.0, -180.0],
            'XPO': [60.0, -60.0, 180.0, -180.0],
        }

        # Mapping of regions to NetCDF mask files
        region_files_map = {
            'CUS': ('/discover/nobackup/projects/gmao/geos_itv/mefrazer/stats/'
                    'masks/CONUS_mask.nc', 'CONUS'),
            'LND': ('/discover/nobackup/projects/gmao/geos_itv/mefrazer/stats/'
                    'masks/land_sea_mask_0.5deg.nc', 'land_mask')
        }

        # Mapping of regions defined as intersections of other regions
        region_intersect_map = {
            'NHL': ['NHE', 'LND'],
            'TRL': ['TRO', 'LND'],
            'SHL': ['SHE', 'LND'],
        }

        # Get target grid
        sample_ds = next(iter(self.datasets.values()))
        target_lats = sample_ds.lat.values
        target_lons = sample_ds.lon.values

        # Process regions (using helper function due to recursive dependencies)
        regional_masks = {}
        processed_regions = set()

        def process_region(region):
            if region in processed_regions:
                return

            # Regions defined as intersections of other regions
            if region in region_intersect_map:
                component_regions = region_intersect_map[region]
                for component in component_regions:
                    # Process prerequisite region if not already processed
                    if component not in processed_regions:
                        process_region(component)
                mask = None
                for component in component_regions:
                    if mask is None:
                        mask = regional_masks[component].copy()
                    else:
                        mask = mask & (regional_masks[component] > 0)
                regional_masks[region] = xr.DataArray(
                    mask.astype(np.int8),
                    coords={'lat': target_lats, 'lon': target_lons},
                    dims=['lat', 'lon'],
                    name=f'{region}_mask'
                )
                processed_regions.add(region)
                return

            # Regions defined by N/S/E/W coords
            if region in local_region_coords_map:
                bounds = local_region_coords_map[region]
                north, south, east, west = bounds
                if west > east:  # Crosses dateline
                    lon_mask = (target_lons >= west) | (target_lons <= east)
                else:
                    lon_mask = (target_lons >= west) & (target_lons <= east)
                lat_mask = (target_lats >= south) & (target_lats <= north)
                mask_2d = np.outer(lat_mask, lon_mask)

            # Regions defined by NetCDF file
            elif region in region_files_map:
                try:
                    file_info = region_files_map[region]
                    if isinstance(file_info, tuple):
                        file_path, var_nms = file_info
                    else:
                        file_path = file_info
                        var_nms = region
                    with xr.open_dataset(file_path,
                                         decode_timedelta=True) as mask_ds:
                        if var_nms not in mask_ds:
                            raise ValueError(f'Variable {var_nms} not found '
                                             f'in NetCDF file for region '
                                             f'{region}')
                        mask_data = mask_ds[var_nms]
                        if (not np.array_equal(mask_data.lat.values,
                                               target_lats) or
                            not np.array_equal(mask_data.lon.values,
                                               target_lons)):
                            mask_2d = self._regrid_mask(mask_data, target_lats,
                                                        target_lons)
                        else:
                            mask_2d = mask_data.values
                except Exception as e:
                    raise ValueError(f'Error processing mask file for region '
                                     f'{region}: {str(e)}')
            else:
                raise ValueError(f'Region {region} is not a valid region')

            # Create regional mask as dataarray
            regional_masks[region] = xr.DataArray(
                mask_2d,
                coords={'lat': target_lats, 'lon': target_lons},
                dims=['lat', 'lon'],
                name=f'{region}_mask'
            )
            processed_regions.add(region)

        for region in requested_regions:
            process_region(region)

        return regional_masks

    @staticmethod
    def save_masks_for_testing(regional_masks: Dict[str, xr.DataArray],
                               output_dir: str = './mask_outputs'):
        '''Save regional masks to text files for debug testing purposes'''

        # Create output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Save each mask after writing
        for region_nm, mask in regional_masks.items():
            output_file = os.path.join(output_dir, f'{region_nm}_mask.txt')
            mask_array = mask.values
            with open(output_file, 'w') as f:
                # Include some metadata as comments
                f.write(f'# Region: {region_nm}\n')
                f.write(f'# Shape: {mask_array.shape}\n')
                f.write(f'# Lat range: {mask.lat.values.min():.2f} '
                        f'to {mask.lat.values.max():.2f}\n')
                f.write(f'# Lon range: {mask.lon.values.min():.2f} to '
                        f'{mask.lon.values.max():.2f}\n')
                f.write(f'# Sum (total 1s): {mask_array.sum()}\n')
                f.write(f'# Fraction covered: '
                        f'{mask_array.sum() / mask_array.size:.4f}\n')
                f.write('#\n')
                # Write the actual mask data
                np.savetxt(f, mask_array, fmt='%d', delimiter=' ')
            print(f'Saved {region_nm} mask to {output_file}')

    @staticmethod
    def _regrid_mask_nearest_neighbor(mask_data: xr.DataArray,
                                      target_lats: np.ndarray,
                                      target_lons: np.ndarray) -> np.ndarray:
        '''
        Regrid a mask using nearest neighbor interpolation.
        Fast method suitable for upscaling (coarse to fine).
        '''

        # Get source coordinates
        source_lats = mask_data.lat.values
        source_lons = mask_data.lon.values

        # Convert source longitudes from 0-360 to -180-180 if needed
        if source_lons.max() > 180:
            source_lons = np.where(source_lons > 180, source_lons - 360,
                                   source_lons)
            lon_sort_idx = np.argsort(source_lons)
            source_lons = source_lons[lon_sort_idx]
            mask_values = mask_data.values[:, lon_sort_idx]
        else:
            mask_values = mask_data.values

        # Handle latitude coordinate ordering
        if source_lats[0] > source_lats[-1]:
            source_lats = source_lats[::-1]
            mask_values = mask_values[::-1, :]

        # Create interpolator (nearest neighbor to preserve 0/1 values)
        interp = RegularGridInterpolator(
            (source_lats, source_lons), mask_values, method='nearest',
            bounds_error=False, fill_value=0)

        # Create target grid coordinates
        target_lat_grid, target_lon_grid = np.meshgrid(
            target_lats, target_lons, indexing='ij')
        points = np.column_stack(
            (target_lat_grid.ravel(), target_lon_grid.ravel()))

        # Interpolate and ensure binary result
        result = interp(points).reshape(target_lat_grid.shape)
        return (result == 1).astype(np.int8)

    @staticmethod
    def _regrid_mask_area_weighted(mask_data: xr.DataArray,
                                   target_lats: np.ndarray,
                                   target_lons: np.ndarray) -> np.ndarray:
        '''
        Regrid a mask using area-weighted majority voting.
        Robust method suitable for downscaling (fine to coarse).
        '''

        # Get source coordinates
        source_lats = mask_data.lat.values
        source_lons = mask_data.lon.values

        # Convert source longitudes from 0-360 to -180-180 if needed
        if source_lons.max() > 180:
            source_lons = np.where(source_lons > 180, source_lons - 360,
                                   source_lons)
            # Sort longitudes and reorder mask accordingly
            lon_sort_idx = np.argsort(source_lons)
            source_lons = source_lons[lon_sort_idx]
            mask_values = mask_data.values[:, lon_sort_idx]
        else:
            mask_values = mask_data.values

        # Handle latitude coordinate ordering
        if source_lats[0] > source_lats[-1]:
            source_lats = source_lats[::-1]
            mask_values = mask_values[::-1, :]

        # Calculate grid spacing (assuming regular grids)
        source_dlat = np.abs(source_lats[1] - source_lats[0]
                             ) if len(source_lats) > 1 else 1.0
        source_dlon = np.abs(source_lons[1] - source_lons[0]
                             ) if len(source_lons) > 1 else 1.0
        target_dlat = np.abs(target_lats[1] - target_lats[0]
                             ) if len(target_lats) > 1 else source_dlat
        target_dlon = np.abs(target_lons[1] - target_lons[0]
                             ) if len(target_lons) > 1 else source_dlon

        # Initialize output array
        result = np.zeros((len(target_lats), len(target_lons)), dtype=np.int8)

        # For each target grid cell
        for i, target_lat in enumerate(target_lats):
            for j, target_lon in enumerate(target_lons):

                # Define target cell boundaries
                lat_min = target_lat - target_dlat / 2
                lat_max = target_lat + target_dlat / 2
                lon_min = target_lon - target_dlon / 2
                lon_max = target_lon + target_dlon / 2

                # Find overlapping source cells
                lat_overlap = (source_lats >= lat_min
                               ) & (source_lats <= lat_max)
                lon_overlap = (source_lons >= lon_min
                               ) & (source_lons <= lon_max)

                # Get overlapping source values
                if np.any(lat_overlap) and np.any(lon_overlap):
                    lat_indices = np.where(lat_overlap)[0]
                    lon_indices = np.where(lon_overlap)[0]

                    # Extract overlapping values
                    overlapping_values = mask_values[np.ix_(lat_indices,
                                                            lon_indices)]

                    # Take majority vote
                    ones_count = np.sum(overlapping_values == 1)
                    zeros_count = np.sum(overlapping_values == 0)

                    # Majority wins (ties go to 0)
                    result[i, j] = 1 if ones_count > zeros_count else 0

        return result

    @staticmethod
    def _regrid_mask(mask_data: xr.DataArray, target_lats: np.ndarray,
                     target_lons: np.ndarray) -> np.ndarray:
        '''
        Regrid a mask using the appropriate method based on resolution:
        nearest neighbor for upscaling, area-weighted majority for downscaling.
        '''

        # Get source grid info
        source_lats = mask_data.lat.values
        source_lons = mask_data.lon.values

        # Calculate approximate grid spacing
        source_dlat = np.abs(source_lats[1] - source_lats[0]
                             ) if len(source_lats) > 1 else 1.0
        source_dlon = np.abs(source_lons[1] - source_lons[0]
                             ) if len(source_lons) > 1 else 1.0
        target_dlat = np.abs(target_lats[1] - target_lats[0]
                             ) if len(target_lats) > 1 else 1.0
        target_dlon = np.abs(target_lons[1] - target_lons[0]
                             ) if len(target_lons) > 1 else 1.0

        # Determine if upscaling or downscaling
        # If target grid is coarser than source, it is downscaling
        is_downscaling = ((target_dlat > source_dlat * 1.1) or
                          (target_dlon > source_dlon * 1.1))
        if is_downscaling:
            print(f'Regridding mask using area-weighted majority '
                  f'(downscaling: {source_dlat:.3f}Â°â†’{target_dlat:.3f}Â°)')
            return StatisticsProcessor._regrid_mask_area_weighted(
                mask_data, target_lats, target_lons)
        else:
            print(f'Regridding mask using nearest neighbor (upscaling: '
                  f'{source_dlat:.3f}Â°â†’{target_dlat:.3f}Â°)')
            return StatisticsProcessor._regrid_mask_nearest_neighbor(
                mask_data, target_lats, target_lons)

    def _initialize_regional_stats_arrays(self):
        '''Initialize output arrays for regional statistics'''

        sample_ds = self.datasets['fcst']
        variables = self.required_vars
        n_init = len(self.init_dates)
        n_leads = len(self.leads)
        n_regions = len(self.regional_masks)
        region_nms = list(self.regional_masks.keys())

        stats_arrays = {}
        for var in variables:
            stats_arrays[var] = {}
            has_levels = 'lev' in sample_ds[var].dims
            # Establish shape/coords/dims
            if has_levels:  # 3d vars
                levels = self.required_levels
                n_levels = len(levels)
                # raw
                raw_shape = (n_init, n_leads, n_levels, n_regions)
                raw_coords = {
                    'init_date': self.init_dates,
                    'lead': self.leads,
                    'lev': levels,
                    'region': region_nms
                }
                raw_dims = ['init_date', 'lead', 'lev', 'region']
                # avg
                avg_shape = (n_leads, n_levels, n_regions)
                avg_coords = {'lead': self.leads, 'lev': levels,
                              'region': region_nms}
                avg_dims = ['lead', 'lev', 'region']
            else:  # 2d vars
                # raw
                raw_shape = (n_init, n_leads, n_regions)
                raw_coords = {
                    'init_date': self.init_dates,
                    'lead': self.leads,
                    'region': region_nms
                }
                raw_dims = ['init_date', 'lead', 'region']
                # avg
                avg_shape = (n_leads, n_regions)
                avg_coords = {'lead': self.leads, 'region': region_nms}
                avg_dims = ['lead', 'region']
            # Create data arrays
            for stat in STATS_TO_CALCULATE_REGIONAL:
                # raw array (init_date as coord)
                stats_arrays[var][stat] = xr.DataArray(
                    np.full(raw_shape, np.nan),
                    coords=raw_coords,
                    dims=raw_dims,
                    name=f'{var}_{stat}'
                )
                # avg array (averaged across init_dates)
                stats_arrays[var][f'{stat}_avg'] = xr.DataArray(
                    np.full(avg_shape, np.nan),
                    coords=avg_coords,
                    dims=avg_dims,
                    name=f'{var}_{stat}_avg'
                )
        return stats_arrays

    def _initialize_global_stats_arrays(self):
        '''Initialize output arrays for global statistics'''
        sample_ds = self.datasets['fcst']
        variables = self.required_vars
        n_init = len(self.init_dates)
        n_leads = len(self.leads)
        n_lat = sample_ds.sizes['lat']
        n_lon = sample_ds.sizes['lon']
        latitude = sample_ds.lat.values
        longitude = sample_ds.lon.values

        stats_arrays = {}
        for var in variables:
            stats_arrays[var] = {}
            has_levels = 'lev' in sample_ds[var].dims
            # Establish shape/coords/dims
            if has_levels:  # 3d vars
                levels = self.required_levels
                n_levels = len(levels)
                # raw
                raw_shape = (n_init, n_leads, n_levels, n_lat, n_lon)
                raw_coords = {'init_date': self.init_dates,
                              'lead': self.leads,
                              'lev': levels,
                              'lat': latitude,
                              'lon': longitude}
                raw_dims = ['init_date', 'lead', 'lev', 'lat', 'lon']
                # avg
                avg_shape = (n_leads, n_levels, n_lat, n_lon)
                avg_coords = {'lead': self.leads,
                              'lev': levels,
                              'lat': latitude,
                              'lon': longitude}
                avg_dims = ['lead', 'lev', 'lat', 'lon']
                # glo
                glo_shape = (n_leads, n_levels)
                glo_coords = {'lead': self.leads, 'lev': levels}
                glo_dims = ['lead', 'lev']
            else:  #2d vars
                # raw
                raw_shape = (n_init, n_leads, n_lat, n_lon)
                raw_coords = {'init_date': self.init_dates,
                              'lead': self.leads,
                              'lat': latitude,
                              'lon': longitude}
                raw_dims = ['init_date', 'lead', 'lat', 'lon']
                # avg
                avg_shape = (n_leads, n_lat, n_lon)
                avg_coords = {'lead': self.leads,
                              'lat': latitude,
                              'lon': longitude}
                avg_dims = ['lead', 'lat', 'lon']
                # glo
                glo_shape = (n_leads,)
                glo_coords = {'lead': self.leads}
                glo_dims = ['lead']
            # Create data arrays
            for stat in STATS_TO_CALCULATE_GLOBAL:
                # raw array (init_date and lat/lon as coord)
                stats_arrays[var][stat] = xr.DataArray(
                    np.full(raw_shape, np.nan),
                    coords=raw_coords,
                    dims=raw_dims,
                    name=f'{var}_{stat}'
                )
                # avg array (averaged across init_dates but lat/lon as coord)
                stats_arrays[var][f'{stat}_avg'] = xr.DataArray(
                    np.full(avg_shape, np.nan),
                    coords=avg_coords,
                    dims=avg_dims,
                    name=f'{var}_{stat}_avg'
                )
                # glo array (averaged across init_dates and lat/lon)
                stats_arrays[var][f'{stat}_glo'] = xr.DataArray(
                    np.full(glo_shape, np.nan),
                    coords=glo_coords,
                    dims=glo_dims,
                    name=f'{var}_{stat}_glo'
                )

        return stats_arrays

    @staticmethod
    def safe_weighted_mean(values, weights, dims=None):
        '''Calculate weighted mean with NaN handling'''
        valid_mask = ~values.isnull() & ~weights.isnull()
        masked_values = values.where(valid_mask, 0)
        masked_weights = weights.where(valid_mask, 0)
        numerator = (masked_values * masked_weights).sum(dim=dims)
        denominator = masked_weights.sum(dim=dims)
        result = numerator / denominator
        return result.where(denominator != 0)

    def _calculate_statistics_loop(self, stats_type):
        '''Main loop to calculate statistics'''

        print('Calculating statistics...')
        # Determine stats to calculate
        if stats_type == 'reg':
            requested_stats = set(STATS_TO_CALCULATE_REGIONAL)
        elif stats_type == 'glo':
            requested_stats = set(STATS_TO_CALCULATE_GLOBAL)
            # Calculate window size for global stats
            sample_var = list(self.stats_arrays.keys())[0]
            lons = self.datasets['fcst'][sample_var].lon
            lon_spacing = float(lons[1] - lons[0])
            window = self._find_window(lon_spacing, 15)
            print(f'Window: {window}, Resulting width: '
                  f'{window * lon_spacing:.1f} degrees longitude')
        print(f'  Calculating statistics: {requested_stats}')
        print(f'  Lead times: {self.leads} hours')

        # Loop over init_dates/leads
        for i, init_date in enumerate(self.init_dates):
            # Print update per init_date
            print(f'  Processing init_date='
                  f'{init_date.strftime("%Y-%m-%d %H:%M")} '
                  f'({i+1}/{len(self.init_dates)})')
            for j, lead in enumerate(self.leads):
                # Get data for this init/lead
                fcst_data = self.datasets['fcst'].sel(init_date=init_date,
                                                      lead=lead)
                valid_time = init_date + timedelta(hours=lead)
                ana_data = self.datasets['ana'].sel(time=valid_time)
                clim_data = self.datasets['clim'].sel(time=valid_time)
                # Regional stats: loop over regions and vars
                if stats_type == 'reg':
                    for k, (region, mask) in enumerate(
                            self.regional_masks.items()):
                        for var in self.stats_arrays.keys():
                            # Apply regional mask
                            fcst_var = fcst_data[var].where(mask)
                            ana_var = ana_data[var].where(mask)
                            clim_var = clim_data[var].where(mask)
                            # Select 3d or 2d weights
                            if 'lev' in fcst_var.dims:
                                weights = self.enhanced_weights['weights_3d'
                                                                ].where(mask)
                            else:
                                weights = self.enhanced_weights['weights_2d'
                                                                ].where(mask)
                            # Calculate statistics and store results
                            stat_results = self._calculate_stats_regional(
                                fcst_var, ana_var, clim_var, weights,
                                requested_stats)
                            for stat_nm, stat_value in stat_results.items():
                                if stat_nm in self.stats_arrays[var]:
                                    if 'lev' in fcst_var.dims:
                                        self.stats_arrays[var][stat_nm][
                                            i, j, :, k] = (stat_value)
                                    else:
                                        self.stats_arrays[var][stat_nm][
                                            i, j, k] = (stat_value)
                # Global stats: loop over vars
                elif stats_type == 'glo':
                    for var in self.stats_arrays.keys():
                        # 3d vars: loop over levels
                        if 'lev' in fcst_data[var].dims:
                            n_levels = fcst_data[var].sizes['lev']
                            for k in range(n_levels):
                                # Select 3d weights
                                weights = self.enhanced_weights['weights_3d'
                                                                ][k, :, :]
                                fcst_var = fcst_data[var][k, :, :]
                                ana_var = ana_data[var][k, :, :]
                                clim_var = clim_data[var][k, :, :]
                                # Calculate statistics and store results
                                stat_results = self._calculate_stats_global(
                                    fcst_var, ana_var, clim_var, weights,
                                    requested_stats, window)
                                for stat_nm, stat_value in (
                                        stat_results.items()):
                                    if stat_nm in self.stats_arrays[var]:
                                        self.stats_arrays[var][stat_nm][
                                            i, j, k, :, :] = (stat_value)
                        # 2d vars
                        else:
                            # Select 2d weights
                            weights = self.enhanced_weights['weights_2d']
                            # Calculate statistics and store results
                            stat_results = self._calculate_stats_global(
                                fcst_data[var], ana_data[var], clim_data[var],
                                weights, requested_stats, window)
                            for stat_nm, stat_value in stat_results.items():
                                if stat_nm in self.stats_arrays[var]:
                                    self.stats_arrays[var][stat_nm][
                                        i, j, :, :] = (stat_value)

    @staticmethod
    def _find_window(lon_spacing, window_goal, min_odd=5, max_odd=21):
        '''
        Find optimal window size for global statistics:
        Selects the in-range odd number that minimizes the difference between
        the actual and goal window sizes (in longitude)
        '''
        odds = range(min_odd, max_odd + 1, 2)
        best_odd = min(odds, key=lambda o: abs(o * lon_spacing - window_goal))
        return best_odd

    def _calculate_stats_regional(self, f, a, c, weights, requested_stats):
        '''Calculate regional statistics with grid weighting'''

        # Calculate means/deviations
        fmean = self.safe_weighted_mean(f, weights, dims=['lat', 'lon'])
        amean = self.safe_weighted_mean(a, weights, dims=['lat', 'lon'])
        cmean = self.safe_weighted_mean(c, weights, dims=['lat', 'lon'])
        fstar = f - fmean
        astar = a - amean
        cstar = c - cmean

        # Mean error, mean squared error, and root mean squared error
        me  = self.safe_weighted_mean(f - a, weights, dims=['lat', 'lon'])
        mse = self.safe_weighted_mean((f - a)**2, weights, dims=['lat', 'lon'])
        rms = mse**(1/2)

        # Calculate variances
        acorr = None
        if 'acorr' in requested_stats:
            fvar = self.safe_weighted_mean((fstar - cstar) * (fstar - cstar),
                                            weights, dims=['lat', 'lon'])
            avar = self.safe_weighted_mean((astar - cstar) * (astar - cstar),
                                            weights, dims=['lat', 'lon'])
            cvar = self.safe_weighted_mean((fstar - cstar) * (astar - cstar),
                                            weights, dims=['lat', 'lon'])
            denominator = fvar**(1/2) * avar**(1/2)
            acorr = (cvar / denominator).where(denominator != 0)


        # Return only requested statistics
        stat_values = {'rms': rms, 'mse': mse, 'me': me, 'acorr': acorr}

        stat_values = {'rms': rms, 'acorr': acorr}
        results = {}
        for stat in requested_stats:
            if stat in stat_values and stat_values[stat] is not None:
                results[stat] = stat_values[stat]

        return results

    def _calculate_stats_global(self, f, a, c, w, requested_stats, n):
        '''Calculate global statistics with rolling windows'''

        # Create rolling windows using xarray
        f_windows = f.rolling(lat=n, lon=n, center=True).construct(
            {'lat': 'lat_window', 'lon': 'lon_window'})
        a_windows = a.rolling(lat=n, lon=n, center=True).construct(
            {'lat': 'lat_window', 'lon': 'lon_window'})
        c_windows = c.rolling(lat=n, lon=n, center=True).construct(
            {'lat': 'lat_window', 'lon': 'lon_window'})
        w_windows = w.rolling(lat=n, lon=n, center=True).construct(
            {'lat': 'lat_window', 'lon': 'lon_window'})

        # Count valid points in each window
        f_valid = (~np.isnan(f_windows)).sum(dim=['lat_window', 'lon_window'])
        a_valid = (~np.isnan(a_windows)).sum(dim=['lat_window', 'lon_window'])
        c_valid = (~np.isnan(c_windows)).sum(dim=['lat_window', 'lon_window'])

        # Create mask for windows with enough valid points (at least 50%)
        min_valid_points = int((n * n) * 0.5)
        valid_mask = (f_valid >= min_valid_points) & (
            a_valid >= min_valid_points) & (c_valid >= min_valid_points)

        # Mean squared error and root mean squared error
        mse = self.safe_weighted_mean(
            (f_windows - a_windows)**2, w_windows,
            dims=['lat_window', 'lon_window'])
        rms = mse**(1/2)
        rms = rms.where(valid_mask)

        # Anomaly correlation (only if requested)
        acorr = None
        if 'acorr' in requested_stats:
            # Calculate means/deviations
            fmean = self.safe_weighted_mean(f_windows, w_windows,
                                            dims=['lat_window', 'lon_window'])
            amean = self.safe_weighted_mean(a_windows, w_windows,
                                            dims=['lat_window', 'lon_window'])
            cmean = self.safe_weighted_mean(c_windows, w_windows,
                                            dims=['lat_window', 'lon_window'])
            fstar_windows = f_windows - fmean
            astar_windows = a_windows - amean
            cstar_windows = c_windows - cmean

            # Calculate variances
            fvar = self.safe_weighted_mean((fstar_windows - cstar_windows) * (
                fstar_windows - cstar_windows), w_windows,
                dims=['lat_window', 'lon_window'])
            avar = self.safe_weighted_mean((astar_windows - cstar_windows) * (
                astar_windows - cstar_windows), w_windows,
                dims=['lat_window', 'lon_window'])
            cvar = self.safe_weighted_mean((fstar_windows - cstar_windows) * (
                astar_windows - cstar_windows), w_windows,
                dims=['lat_window', 'lon_window'])

            denominator = fvar**(1/2) * avar**(1/2)
            acorr = (cvar / denominator).where(denominator != 0)
            acorr = acorr.where(valid_mask)

        # Return only requested statistics
        stat_values = {'rms': rms, 'acorr': acorr, 'f': f}
        results = {}
        for stat in requested_stats:
            if stat in stat_values and stat_values[stat] is not None:
                results[stat] = stat_values[stat]

        return results

    def _compute_averaged_statistics(self):
        '''Compute averaged statistics (over init_dates) from raw statistics'''
        print('Computing averaged statistics...')

        for var in self.stats_arrays.keys():
            base_stats = [key for key in self.stats_arrays[var].keys() if not
                          key.endswith('_avg') and not key.endswith('_glo')]
            for stat_nm in base_stats:
                avg_key = f'{stat_nm}_avg'
                if avg_key in self.stats_arrays[var]:
                    self.stats_arrays[var][avg_key] = self.stats_arrays[var][
                        stat_nm].mean(dim='init_date')

    def _compute_global_avg_statistic(self):
        '''Compute properly weighted global average statistics from average'''
        print('Computing global average statistics...')

        for var in self.stats_arrays.keys():
            base_stats = [key for key in self.stats_arrays[var].keys()
                          if key.endswith('_avg')]
            if 'lev' in self.stats_arrays[var][base_stats[0]].dims:
                weights = self.enhanced_weights['weights_3d']
            else:
                weights = self.enhanced_weights['weights_2d']
            for stat_nm in base_stats:
                glo_key = f'{stat_nm.replace("avg", "glo")}'
                if glo_key in self.stats_arrays[var]:
                    self.stats_arrays[var][glo_key] = self.safe_weighted_mean(
                        self.stats_arrays[var][stat_nm], weights,
                        dims=['lat', 'lon'])

    def _save_statistics_output(
            self, stats_type, info_dir=None, using_chunks=False,
            chunk_number=None, skip_avg=False):
        '''Save statistics to NetCDF file with metadata (if not skip_avg)'''
        print('Saving statistics output...')

        # Save each variable with base_stat's units/long_name in one dataset
        all_stats = {}
        for var in self.stats_arrays.keys():
            var_upper = var.upper()
            var_unit = VARS_UNIT_MAP.get(var_upper, 'unknown')
            var_long_nm = VARS_LONG_MAP.get(var_upper)
            for stat_key, stat_array in self.stats_arrays[var].items():
                # Skip avg and glo variables if skip_avg
                if skip_avg and (stat_key.endswith('_avg')
                                 or stat_key.endswith('_glo')):
                    continue
                # Create a copy of the array to add attributes
                output_array = stat_array.copy(deep=True)
                # Get base stat name
                base_stat = stat_key
                if base_stat.endswith('_avg') or base_stat.endswith('_glo'):
                    base_stat = base_stat[:-4]
                # Add long_name attribute
                if base_stat in STATS_LONG_NAMES:
                    stat_long_nm = (
                        f'{STATS_LONG_NAMES[base_stat]} of {var_long_nm}')
                    output_array.attrs['long_name'] = stat_long_nm
                # Add units attribute
                if base_stat in UNITLESS_STATS:
                    output_array.attrs['units'] = 'unitless'
                elif base_stat in SQUARED_UNIT_STATS:
                    output_array.attrs['units'] = f'{var_unit}Â²'
                else:
                    output_array.attrs['units'] = var_unit
                all_stats[f'{var}_{stat_key}'] = output_array

        # Include grid_weights if global
        if stats_type == 'glo':
            all_stats['grid_weights'] = self.datasets['fcst'][
                'grid_weights'].astype('float32')

        # Create final dataset
        stats_ds = xr.Dataset(all_stats)

        # Add global metadata
        if stats_type == 'reg':
            title = 'Regional Forecast Verification Statistics'
            description = 'Regional forecast verification statistics'
        elif stats_type == 'glo':
            title = 'Global Forecast Verification Statistics'
            description = 'Global forecast verification statistics'
        fcst_model = self.config.get('fcst_model', '')
        ana_model = self.config.get('ana_model', '')
        clim_model = self.config.get('clim_model', '')
        stats_ds.attrs.update({
            'title': title,
            'created': datetime.now().isoformat(),
            'description': description,
            'forecast_model': fcst_model,
            'analysis_model': ana_model,
            'climatology_model': clim_model,
        })

        # Add exclusion information if present
        BatchDatasetProcessor._add_exclusion_metadata(stats_ds, self.config)

        # Generate final output filename
        if hasattr(self, 'save_dir') and self.save_dir:  # Single-forecast mode
            date_str = str(self.config['start_date'])
            fdays = self.config['FDAYS']
            interval = self.config['fcst_interval']
            nlat = self.config['Nlat']
            nlon = self.config['Nlon']
            suffix = f'{date_str}_len{fdays}d_int{interval}h_{nlat}x{nlon}.nc4'
            final_output_file = os.path.join(
                self.save_dir, (f'stats_regional_{fcst_model}_{ana_model}_'
                                f'{clim_model}_{suffix}'))
        else:  # Regular mode
            fcst_filenm = self.dataset_files.get('fcst')
            suffix = os.path.basename(fcst_filenm).split(
                f'{fcst_model}_', 1)[1]
            if stats_type == 'reg':
                final_output_file = (f'outputs/stats_regional_{fcst_model}_'
                                   f'{ana_model}_{clim_model}_{suffix}')
            elif stats_type == 'glo':
                final_output_file = (f'outputs/stats_global_{fcst_model}_'
                                   f'{ana_model}_{clim_model}_{suffix}')

        stats_ds.attrs['final_output_filenm'] = final_output_file
        # For chunks, create chunk filename (final name is saved for merging)
        if using_chunks:
            output_file = (f'outputs/{info_dir}/tmp/stats_chunk_{stats_type}_'
                           f'{chunk_number}.nc4')
        # For regular, save to final name
        else:
            output_file = final_output_file

        # Save with compression
        encoding = {}
        for var in stats_ds.variables:
            if var == 'region':
                continue
            encoding[var] = {'zlib': True, 'complevel': 2}
            dtype_nm = str(stats_ds[var].dtype)
            if 'float64' in dtype_nm:
                encoding[var]['dtype'] = 'float32'
        stats_ds.to_netcdf(output_file, encoding=encoding)
        print(f'[OK] Statistics saved to: {output_file}')

        # Print variable names and sample dimensions
        var_nms = list(all_stats.keys())
        print(f'    Statistics: {len(var_nms)} total')
        print(f'    Variables: {var_nms}')
        if all_stats:
            first_stat = next(iter(all_stats.values()))
            print(f'    Dimensions: {dict(first_stat.sizes)}')

        # Write scorecard files for regional stats
        if stats_type == 'reg':
            self._write_scorecard_files()

        return stats_ds

    def _write_scorecard_files(self):
        '''Write scorecard files - one per init_date'''
        print('\n==================================================')
        print('WRITING SCORECARD FILES')
        print('==================================================')

        # Get config values
        model      = self.config.get('fcst_model', '')
        ana_model  = self.config.get('ana_model', '')
        clim_model = self.config.get('clim_model', '')
        expver     = self.config.get('expver', '')
        verify     = self.config.get('verify', '')
        start_date = self.config['start_date']
        end_date   = self.config['end_date']
        date_range = f'{start_date}-{end_date}'
        fvars = {
            'de3d': self.config.get('3d_vars_default', []),
            'de2d': self.config.get('2d_vars_default', []),
            'sl2d': self.config.get('2d_vars_slices',  []),
            'ae2d': self.config.get('2d_vars_aerosol', [])
        }
        collections = [coll for coll, vars_list in fvars.items() if vars_list]
        is_3d = {coll: (int(coll[-2]) == 3) for coll in collections}

        # Create scorecard_input directory
        if hasattr(self, 'save_dir') and self.save_dir:  # Single-forecast mode
            scorecard_dir = self.save_dir
        else:  # Regular mode
            scorecard_dir = (f'outputs/scorecard_input_{model}_{ana_model}_'
                             f'{clim_model}_{date_range}')
        os.makedirs(scorecard_dir, exist_ok=True)

        # Statistics mapping
        stat_mapping = {'acorr': 'cor', 'rms': 'rms', 'rms_ran': 'rms_ran',
            'rms_bar': 'rms_bar', 'rms_dis': 'rms_dis', 'rms_dsp': 'rms_dsp'}

        def reverse_scaling(var, stat_key, value):
            '''Helper: revert Q and PM vars to SI units for scorecards'''
            if var.upper() in ['Q', 'Q2M'] and stat_key != 'acorr':
                if stat_key in ['rms_dis', 'rms_dsp']:  # squared stats
                    return value * 1E-6
                else:
                    return value * 1E-3
            if var.upper() == 'PM25' and stat_key != 'acorr':
                if stat_key in ['rms_dis', 'rms_dsp']:  # squared stats
                    return value * 1E-12
                else:
                    return value * 1E-6
            return value

        def write_line(file, pressure, var, stat_nm, lead, value, date_str,
                       domain_nm, north, south, east, west):
            '''Helper: write a single scorecard line'''
            def f7(n):
                '''Helper: format number to 7 significant figures'''
                if np.isnan(n): return '9.9999999E+14'
                elif n == 0.0: return '0.0000000'
                else:
                    abs_n = abs(n)
                    if abs_n < 0.01 or abs_n >= 10000:
                        return f'{n:.6E}'
                    else:
                        int_part_len = len(str(abs(int(n))))
                        decimals = max(0, 7 - int_part_len)
                        format_str = f'{{:.{decimals}f}}'
                        return format_str.format(n)
            # Write formatted line
            file.write(f'0.0|{date_str}|{domain_nm}|{f7(east)}|{model}|'
                       f'{expver}|{f7(pressure)}|pl|{f7(north)}|{var.lower()}|'
                       f'{expver}|{f7(south)}|{stat_nm}|{lead}|fc|'
                       f'{f7(value)}|{verify}|{f7(west)}|\n')

        def process_stats_for_var(
                file, var, pressure, lev_idx, date_str, domain_nm, north,
                south, east, west, init_idx, lead_idx, region_idx):
            '''Helper: process/write all stats for a variable (and level)'''
            for stat_key, stat_nm in stat_mapping.items():
                if stat_key in self.stats_arrays[var]:
                    stat_array = self.stats_arrays[var][stat_key]
                    # Extract value based on dimensionality
                    if lev_idx is not None:  # 3D variable
                        value = float(stat_array.isel(
                            init_date=init_idx, lead=lead_idx, lev=lev_idx,
                            region=region_idx).values)
                    else:  # 2D variable
                        value = float(stat_array.isel(
                            init_date=init_idx, lead=lead_idx,
                            region=region_idx).values)
                    # Apply Q scaling and write line
                    value = reverse_scaling(var, stat_key, value)
                    write_line(file, pressure, var, stat_nm,
                               self.leads[lead_idx], value, date_str,
                               domain_nm, north, south, east, west)

        # Header line
        header = ('count|date|domain_name|east|expver|forecast|level|levtype|'
                  'north|variable|source|south|statistic|step|type|value|'
                  'verify|west')

        # Create file for each init_date
        for i, init_date in enumerate(self.init_dates):
            file_date = init_date.strftime('%Y%m%d_%Hz')
            filenm = (f'{scorecard_dir}/{model}.{verify}.fstat.stds.log.'
                      f'{file_date}.txt')
            date_str = init_date.strftime('%Y%m%d%H')
            with open(filenm, 'w') as file:
                file.write(f'{header}\n')
                # Loop by leads and regions
                for j, lead in enumerate(self.leads):
                    for region_nm in self.regional_masks.keys():
                        if region_nm not in REGION_SHORT_MAP:
                            print(f'[ERROR] Region "{region_nm}" not found in '
                                  f'REGION_SHORT_MAP')
                            sys.exit(1)
                        # Get region info to include in scorecard lines
                        if region_nm in REGION_COORDS_MAP:
                            north, south, east, west = REGION_COORDS_MAP[
                                region_nm]
                        else:
                            north, south, east, west = 0.0, 0.0, 0.0, 0.0
                        domain_nm = REGION_SHORT_MAP[region_nm]
                        region_idx = list(self.regional_masks.keys()).index(
                            region_nm)
                        # Process stats by collection/variable
                        for coll in fvars.keys():
                            if fvars[coll] is not None:
                                for var in fvars[coll]:
                                    if var in self.stats_arrays:
                                        if is_3d[coll]:
                                            # 3D variables
                                            pressure_levels = (
                                                self.stats_arrays[var][
                                                    'rms'].lev.values)
                                            for k, pressure in enumerate(
                                                    pressure_levels):
                                                process_stats_for_var(
                                                    file, var, pressure, k,
                                                    date_str, domain_nm, north,
                                                    south, east, west, i, j,
                                                    region_idx)
                                        else:
                                            # 2D variables
                                            pressure = (1000.0 if coll[:2]
                                                        == 'de' else 0.0)
                                            process_stats_for_var(
                                                file, var, pressure, None,
                                                date_str, domain_nm, north,
                                                south, east, west, i, j,
                                                region_idx)

            print(f'    [OK] Wrote {filenm}')
        print(f'[SUCCESS] Wrote {len(self.init_dates)} scorecard files')

    @classmethod
    def merge_statistics_files(cls, stats_type: str, info_dir: str) -> bool:
        '''Merge chunked stats files and calculate averages over init_dates'''
        print('\n==================================================')
        print('MERGING STATISTICS FILES')
        print('==================================================')

        # Establish filename pattern, look for chunks, and report
        pattern = f'outputs/{info_dir}/tmp/stats_chunk_{stats_type}_*.nc4'
        print(f'[INFO] Looking for chunks for {stats_type}')
        chunk_files = sorted(glob.glob(pattern))
        if not chunk_files:
            print(f'[ERROR] No statistics files found matching pattern: '
                  f'{pattern}')
            return False
        print(f'[INFO] Found {len(chunk_files)} statistics files to merge:')
        for file in chunk_files:
            print(f'  - {file}')

        # Load and merge, closing and cleaning after each file
        print('[INFO] Loading and merging statistics files sequentially...')
        merged_ds = None
        output_file = None
        for i, file in enumerate(chunk_files):
            try:
                print(f'[INFO] Processing file {i+1}/{len(chunk_files)}: '
                      f'{os.path.basename(file)}')
                current_ds = xr.open_dataset(file, decode_timedelta=True)
                if output_file is None:
                    output_file = current_ds.attrs.get('final_output_filenm')
                    if output_file is None:
                        raise ValueError(f'No final_output_filenm attribute '
                                         f'found in chunk file {file}. '
                                         f'Cannot save merged results.')
                    print(f'[INFO] Merged statistics will be saved to: '
                          f'{output_file}')
                if merged_ds is None:
                    merged_ds = current_ds
                    if stats_type == 'glo' and 'grid_weights' in merged_ds:
                        original_grid_weights = merged_ds['grid_weights'
                                                          ].copy(deep=True)
                    print(f'[INFO] Loaded base dataset with '
                          f'{len(merged_ds.init_date)} init dates')
                else:
                    merged_ds = xr.concat([merged_ds, current_ds],
                                          dim='init_date')
                    print(f'[INFO] Merged dataset now has '
                          f'{len(merged_ds.init_date)} init dates')
                    current_ds.close()
                    del current_ds
                gc.collect()
            except Exception as e:
                print(f'[WARNING] Could not process {file}: {e}')
                continue
        if merged_ds is None:
            print('[ERROR] No valid statistics datasets could be loaded')
            return False
        print(f'[INFO] Successfully merged {len(chunk_files)} datasets')

        try:
            # Sort merged dataset by init_date
            merged_ds = merged_ds.sortby('init_date')
            print(f'[INFO] Sorted {len(merged_ds.init_date)} init dates in '
                  f'chronological order')

            # restore regular grid weights if global
            if stats_type == 'glo' and 'original_grid_weights' in locals():
                if 'grid_weights' in merged_ds:
                    print('[INFO] Restoring original 2D grid_weights')
                    merged_ds['grid_weights'] = original_grid_weights

            # Create enhanced (NaN-masked) weights if global
            if stats_type == 'glo':
                enhanced_weights = None
                print('[INFO] Creating enhanced grid weights with NaN '
                      'masking...')
                temp_ds = xr.Dataset(
                    {var: merged_ds[var] for var in merged_ds.data_vars
                     if var != 'grid_weights'})
                dummy_datasets = {'fcst': temp_ds}
                base_weights = merged_ds['grid_weights']
                # Establish sample #d and 2d vars
                sample_3d_var = None
                sample_2d_var = None
                for var_nms in temp_ds.data_vars:
                    var = temp_ds[var_nms]
                    if 'lev' in var.dims and sample_3d_var is None:
                        sample_3d_var = var_nms
                    elif 'lev' not in var.dims and sample_2d_var is None:
                        sample_2d_var = var_nms
                    if sample_3d_var and sample_2d_var:
                        break
                # Simplified enhanced weights creation for merging
                enhanced_weights = {}
                if sample_3d_var:
                    sample_var = temp_ds[sample_3d_var]
                    nan_mask_3d = sample_var.isnull().any(dim='init_date')
                    levels = nan_mask_3d.lev
                    weights_3d = base_weights.broadcast_like(nan_mask_3d)
                    weights_3d = weights_3d.where(~nan_mask_3d)
                    enhanced_weights['weights_3d'] = weights_3d
                if sample_2d_var:
                    sample_var = temp_ds[sample_2d_var]
                    nan_mask_2d = sample_var.isnull().any(dim='init_date')
                    weights_2d = base_weights.where(~nan_mask_2d)
                    enhanced_weights['weights_2d'] = weights_2d
                print('[INFO] Successfully created enhanced weights')

            # Add variable metadata and averaged variables
            if stats_type == 'glo':
                print('[INFO] Calculating time and global spatial averages...')
            else:
                print('[INFO] Calculating time averages...')
            for var in list(merged_ds.data_vars):
                # For vars with init_date as a dim
                if 'init_date' in merged_ds[var].dims:
                    print(f'Processing {var}...')
                    base_stat = var.split('_', 1)[1]
                    var_upper = (var.split('_', 1)[0]).upper()
                    # Include units and long_names
                    var_unit = VARS_UNIT_MAP.get(var_upper, 'unknown')
                    var_long_nm = VARS_LONG_MAP.get(var_upper)
                    if base_stat in STATS_LONG_NAMES:
                        stat_long_nm = (f'{STATS_LONG_NAMES[base_stat]} of '
                                        f'{var_long_nm}')
                    else:
                        stat_long_nm = f'Statistic of {var_long_nm}'
                    if base_stat in UNITLESS_STATS:
                        stat_unit = 'unitless'
                    elif base_stat in SQUARED_UNIT_STATS:
                        stat_unit = f'{var_unit}Â²'
                    else:
                        stat_unit = var_unit
                    # Create avg (over init_dates) variable
                    avg_var = f'{var}_avg'
                    merged_ds[avg_var] = merged_ds[var].mean(
                        dim='init_date').astype('float32')
                    merged_ds[avg_var].attrs.update({
                        'long_name': stat_long_nm,
                        'units': stat_unit
                    })
                    # Add global spatial average if global stats
                    if stats_type == 'glo':
                        glo_var = f'{var}_glo'

                        if 'lev' in merged_ds[avg_var].dims:
                            weights = enhanced_weights['weights_3d']
                        else:
                            weights = enhanced_weights['weights_2d']
                        merged_ds[glo_var] = cls.safe_weighted_mean(
                            merged_ds[avg_var], weights, dims=['lat', 'lon']
                        ).astype('float32')
                        merged_ds[glo_var].attrs.update({
                            'long_name': stat_long_nm,
                            'units': stat_unit
                        })

            # Add merge-related global attributes and save with compression
            final_ds = merged_ds
            final_ds.attrs['merge_date'] = datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S')
            final_ds.attrs['num_chunks_merged'] = len(chunk_files)
            final_ds.attrs['chunk_files'] = ', '.join([os.path.basename(f)
                                                       for f in chunk_files])
            print(f'[INFO] Saving merged statistics to {output_file}...')
            encoding = {}
            for var in final_ds.variables:
                if var == 'region':
                    continue
                dtype_nm = str(final_ds[var].dtype)
                if 'float64' in dtype_nm:
                    encoding[var] = {'dtype': 'float32'}
            final_ds.to_netcdf(output_file, encoding=encoding)
            print(f'[SUCCESS] Merged statistics saved to {output_file}')

            # Delete chunk files (after successful merge)
            print('[INFO] Deleting chunk files after successful merge...')
            deleted_count = 0
            for file in chunk_files:
                try:
                    os.remove(file)
                    deleted_count += 1
                    print(f'  [OK] Deleted: {file}')
                except Exception as e:
                    print(f'  [ERROR] Could not delete {file}: {e}')
            print(f'[INFO] Deleted {deleted_count}/{len(chunk_files)} chunk '
                  'files')
            return True
        except Exception as e:
            print(f'[ERROR] Failed to merge statistics: {e}')
            traceback.print_exc()
            return False

