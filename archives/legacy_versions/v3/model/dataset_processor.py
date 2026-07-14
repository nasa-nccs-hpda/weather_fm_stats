'''Dataset processing model.'''

import calendar
import gc
import glob
import os
import re
import traceback
import yaml
from datetime import datetime, timedelta
from typing import List, Dict, Any

import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from metpy.units import units
from scipy.interpolate import RegularGridInterpolator

from model.config_model import DuplicateKeysError, SafeLoaderWithDuplicateKeyCheck
from model.constants import *

# ================== DATASET PROCESSING CLASS ==================


class BatchDatasetProcessor:

    def __init__(self, config: Dict[str, Any]):
        '''Initialize with config'''
        if config is None:
            raise ValueError('Config dictionary is required')
        self.config = config
        missing = 'Missing required config parameter:'

        # Store model/analysis names (required)
        try:
            self.fcst_model = config['fcst_model']
        except KeyError:
            raise ValueError(f'{missing} fcst_model')
        try:
            self.ana_model = config['ana_model']
        except KeyError:
            raise ValueError(f'{missing} ana_model')
        try:
            self.clim_model = config['clim_model']
        except KeyError:
            raise ValueError(f'{missing} clim_model')
        try:
            self.expver = config['expver']
        except KeyError:
            raise ValueError(f'{missing} expver (needed for scorecards)')
        try:
            self.verify = config['verify']
        except KeyError:
            raise ValueError(f'{missing} verify (needed for scorecards)')

        # Store pressure levels and regions
        try:
            self.pressure_levels = config['levels']
        except KeyError:
            raise ValueError(f'{missing} levels')
        try:
            self.regions = config['regions']
        except KeyError:
            raise ValueError(f'{missing} regions')

        # Store target grid settings
        try:
            self.nlat = config['Nlat']
        except KeyError:
            raise ValueError(f'{missing} Nlat')
        try:
            self.nlon = config['Nlon']
        except KeyError:
            raise ValueError(f'{missing} Nlon')
        try:
            self.lat_res = config['delta_lat']
        except KeyError:
            raise ValueError(f'{missing} delta_lat')
        try:
            self.lon_res = config['delta_lon']
        except KeyError:
            raise ValueError(f'{missing} delta_lon')
        self.target_grid = self._create_target_grid()

        # Store base paths (required)
        try:
            self.fcst_dir = config['fcst_input_dir']
        except KeyError:
            raise ValueError(f'{missing} fcst_input_dir')
        try:
            self.ana_dir = config['ana_input_dir']
        except KeyError:
            raise ValueError(f'{missing} ana_input_dir')
        try:
            self.clim_dir = config['clim_input_dir']
        except KeyError:
            raise ValueError(f'{missing} clim_input_dir')

        # Parse collection-specific directories (SANDY)
        self.dir_colls = self._parse_dir_colls(config)

        # Parse variable collections (saving original before filtering)
        self.var_colls = self._parse_var_colls(config)
        self.original_var_colls = self.var_colls.copy()
        self.all_vars_3d, self.all_vars_2d = self._get_all_variables()

        # Parse variable aliases (case-insensitive)
        self.var_aliases = self._parse_variable_aliases(config)

        # Parse template collections
        self.template_colls = self._parse_template_colls(config)

        # Initialize regridder cache
        self.regridders = {}

        # Process search directories (ensure it is a list)
        dir_loc = config.get('dir_loc', [])
        if dir_loc is None:
            # Handle blank or missing entry
            search_dirs = []
        elif isinstance(dir_loc, str):
            # Split comma-separated string into list
            search_dirs = [d.strip() for d in dir_loc.split(',') if d.strip()]
        else:
            # Already a list, just clean it up
            search_dirs = [str(d).strip() for d in dir_loc if str(d).strip()]

        # Always add cwd/output to search locations
        output_dir = os.path.join(os.getcwd(), 'outputs')
        if output_dir not in search_dirs:
            search_dirs.append(output_dir)

        # Store the processed search directories back in config
        self.config['dir_loc'] = search_dirs

        # Print concise initialization summary
        print('\nInitialized BatchDatasetProcessor from YAML config')
        print(f'Models: Forecast={self.fcst_model}, Analysis={self.ana_model},'
              f' Climatology={self.clim_model}')
        print(f'Target pressure levels: {self.pressure_levels}')

        # Print collection information
        coll_nms = list(self.var_colls.keys())
        if coll_nms:
            print(f'Variable collections: {coll_nms}')
            for coll, vars_dict in self.var_colls.items():
                if vars_dict['3d'] or vars_dict['2d']:
                    print(f'  {coll}: 3D={vars_dict["3d"]}, '
                          f'2D={vars_dict["2d"]}')

    def _parse_var_colls(self, config: Dict[str, Any]
                                    ) -> Dict[str, Dict[str, List[str]]]:
        '''Parse variable collections from config'''
        collections = {}

        # Find all variable collection keys
        for key in config.keys():
            # Check for any of the known collection names
            collection_check = any(key.endswith(
                f'_vars_{coll}') for coll in ALL_COLL_NMS)
            if collection_check:
                # Extract dimension and collection name
                parts = key.split('_vars_')
                if len(parts) == 2:
                    dim_str, coll = parts
                    dimension = dim_str  # '2d' or '3d'
                    if coll not in collections:
                        collections[coll] = {'3d': [], '2d': []}
                    # Get variables and normalize names
                    raw_vars = config.get(key, [])
                    if raw_vars:
                        normalized_vars = [self._normalize_variable_name(var)
                                           for var in raw_vars]
                        collections[coll][dimension] = normalized_vars
        return collections

    def _normalize_variable_name(self, var_nms: str) -> str:
        '''Normalize variable name according to naming convention'''
        var_upper = var_nms.upper()
        # Find first digit
        first_digit_pos = -1
        for i, char in enumerate(var_upper):
            if char.isdigit():
                first_digit_pos = i
                break
        if first_digit_pos == -1:
            # All caps when no digits
            return var_upper
        else:
            # With digits, caps before and lowercase after
            return (var_upper[:first_digit_pos]
                    + var_upper[first_digit_pos:].lower())

    def _get_all_variables(self) -> tuple:
        '''Get all unique variables across collections'''
        all_3d = set()
        all_2d = set()
        for coll_vars in self.var_colls.values():
            all_3d.update(coll_vars['3d'])
            all_2d.update(coll_vars['2d'])
        return sorted(list(all_3d)), sorted(list(all_2d))

    def _parse_variable_aliases(self, config: Dict[str, Any]
                                ) -> Dict[str, List[str]]:
        '''Parse variable aliases from config'''
        aliases = {}
        # Find all alias keys in config (keys ending with '_alias')
        for key in config.keys():
            if key.endswith('_alias'):
                # Extract variable name (remove '_alias' suffix)
                var_nms = key[:-6]
                # Make all aliases lowercase for case-insensitive matching
                aliases[var_nms] = [alias.lower() for alias in config[key]]
        return aliases

    def _parse_template_colls(self, config: Dict[str, Any]
                                    ) -> Dict[str, Dict[str, List[str]]]:
        '''Parse template collections from config'''
        collections = {}
        for dataset_type in ['fcst', 'ana', 'clim']:
            collections[dataset_type] = {}
            # Find all template keys for this dataset type
            template_prefix = f'{dataset_type}_input_template_'
            existing_collections = set()
            for key in config.keys():
                if key.startswith(template_prefix):
                    coll_nm = key[len(template_prefix):]
                    template_value = config.get(key, '')
                    if template_value and template_value.strip():
                        # Split on commas (excepting shift syntax)
                        if isinstance(template_value, str):
                            parts = []
                            current_part = ''
                            brace_depth = 0
                            for char in template_value:
                                if char == '{':
                                    brace_depth += 1
                                elif char == '}':
                                    brace_depth -= 1
                                elif char == ',' and brace_depth == 0:
                                    # Comma outside of braces: split here
                                    if current_part.strip():
                                        parts.append(current_part.strip())
                                    current_part = ''
                                    continue
                                current_part += char
                            # Add the last part
                            if current_part.strip():
                                parts.append(current_part.strip())
                            templates = parts if parts else [
                                template_value.strip()]
                        else:
                            templates = template_value if isinstance(
                                template_value, list) else [template_value]
                        collections[dataset_type][coll_nm] = templates
                        existing_collections.add(coll_nm)
                    else:
                        # Empty or None: create empty entry for fallback
                        collections[dataset_type][coll_nm] = []
                        existing_collections.add(coll_nm)
            # Ensure all collections exist for fallback logic
            for coll_nm in ALL_COLL_NMS:
                if coll_nm not in existing_collections:
                    collections[dataset_type][coll_nm] = []
        return collections

    def _parse_dir_colls(self, config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        '''Parse collection-specific directory paths from config'''
        collections = {}
        for dataset_type in ['fcst', 'ana', 'clim']:
            collections[dataset_type] = {}
            # Find all directory keys for this dataset type
            dir_prefix = f'{dataset_type}_input_dir_'
            for key in config.keys():
                if key.startswith(dir_prefix):
                    coll_nm = key[len(dir_prefix):]
                    dir_value = config.get(key, '')
                    if dir_value and dir_value.strip():
                        collections[dataset_type][coll_nm] = dir_value.strip()
        return collections

    @classmethod
    def from_yaml(cls, yaml_file, single_fcst_mode=None):
        '''Create BatchDatasetProcessor from YAML config file'''
        # Open YAML
        print(f'Loading configuration from: {yaml_file}')
        try:
            with open(yaml_file, 'r') as f:
                yaml_config = yaml.load(
                    f, Loader=SafeLoaderWithDuplicateKeyCheck)
        except DuplicateKeysError as e:
            raise ValueError(str(e))
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f'Error reading YAML file: {e}')
        # Convert YAML structure to internal format
        config = cls._convert_yaml_config(yaml_config, single_fcst_mode)
        return cls(config=config)

    @staticmethod
    def _convert_yaml_config(yaml_config, single_fcst_mode=None):
        '''Convert YAML config structure to internal format'''
        params = yaml_config['parameters']

        # Validate if single-forecast mode
        if single_fcst_mode is not None:
            override_date = single_fcst_mode
            datetime.strptime(override_date, '%Y%m%d')  # Validate format
            single_fcst_date = int(override_date)

        # Convert interval from hours to HHMMSS format
        interval_hours = params['fcst_interval']
        nfreq = f'{interval_hours:02d}0000'

        if not single_fcst_mode:
            # Convert date range
            start_date = datetime.strptime(str(params['start_date']), '%Y%m%d')
            end_date = datetime.strptime(str(params['end_date']), '%Y%m%d')
            fdates = (
                f'{start_date.strftime("%Y%m%d")}-'
                f'{end_date.strftime("%Y%m%d")}')
        else:
            fdates = f'{single_fcst_date}-{single_fcst_date}'

        # Validate regions
        available_regions = {
            'GLO', 'NHE', 'TRO', 'SHE', 'NWQ', 'NEQ', 'SWQ', 'SEQ', 'NAM',
            'EUR', 'NPO', 'SPO', 'XPO', 'CUS', 'LND', 'NHL', 'SHL', 'TRL'
        }
        invalid_regions = [
            r for r in params['regions'] if r not in available_regions]
        if invalid_regions:
            raise ValueError(f'Invalid regions requested: {invalid_regions}. '
                             f'Available regions: {sorted(available_regions)}')

        # Add date/time parameters
        config = {
            'FDAYS': params['fcst_length'],
            'FDATES': fdates,
            'NFREQ': nfreq,
        }

        if not single_fcst_mode:
            config.update({
                'start_date': params['start_date'],
                'end_date': params['end_date'],
                'fcst_spacing': params.get('fcst_spacing', 1),
                'exclude_dates': params.get('exclude_dates', []),
            })
        else:
            config.update({
                'start_date': single_fcst_date,
                'end_date': single_fcst_date,
                'fcst_spacing': 1,
                'exclude_dates': [],
            })

        # Add all other parameters
        for key, value in params.items():
            if key not in config:
                # Sort pressure levels in descending order (surface to top)
                if key == 'levels':
                    config[key] = sorted(value, reverse=True)
                # Clean comments from variable lists
                elif key.startswith(('2d_vars_', '3d_vars_')
                                    ) and isinstance(value, list):
                    # Clean each variable name from potential comments
                    cleaned_vars = []
                    for var in value:
                        # If the variable is a string, strip any comments
                        if isinstance(var, str):
                            # Split on '#' and take only the first part
                            var_nms = var.split('#')[0].strip()
                            if var_nms:
                                cleaned_vars.append(var_nms)
                        else:
                            cleaned_vars.append(var)
                    config[key] = cleaned_vars
                else:
                    config[key] = value
        return config

    def _create_target_grid(self):
        '''Create target grid for regridding with compatibility checking'''
        nlat = self.nlat
        nlon = self.nlon
        delta_lat = self.lat_res
        delta_lon = self.lon_res
        # Check grid compatibility
        expected_lat_span = (nlat - 1) * delta_lat
        expected_lon_span = nlon * delta_lon
        # For lat: should span from -90 to +90 (180 degrees total)
        if abs(expected_lat_span - 180.0) > 0.001:
            raise ValueError(f'Grid incompatibility: Nlat={nlat} with '
                             f'delta_lat={delta_lat} gives span of '
                             f'{expected_lat_span}Â°, but need exactly 180Â°')
        # For lon: should span 360 degrees
        if abs(expected_lon_span - 360.0) > 0.001:
            raise ValueError(f'Grid incompatibility: Nlon={nlon} with '
                             f'delta_lon={delta_lon} gives span of '
                             f'{expected_lon_span}Â°, but need exactly 360Â°')
        # Create latitude: -90 to +90 with poles as partial cells
        lat = np.linspace(-90, 90, nlat)
        # Create longitude: -180 to +180 (exclusive)
        lon = np.linspace(-180, 180, nlon, endpoint=False)
        # Create data vars
        ds_out = xr.Dataset({
            'lat': (['lat'], lat, {'units': 'degrees_north'}),
            'lon': (['lon'], lon, {'units': 'degrees_east'}),
        })
        # Report results
        print('\nCreated target grid:')
        print(f'  Latitude: {nlat} points, {lat[0]:.1f}Â° to {lat[-1]:.1f}Â°, '
              f'spacing {delta_lat:.1f}Â°')
        print(f'  Longitude: {nlon} points, {lon[0]:.1f}Â° to {lon[-1]:.1f}Â°, '
              f'spacing {delta_lon:.1f}Â°')
        print('  Note: Poles treated as partial cells (-90Â° = -90 to -89Â°, '
              '+90Â° = +89 to +90Â°)')
        # Assign target grid and calculate grid weights
        self.target_grid = ds_out
        self.grid_weights = self._calculate_latitude_weights()
        return ds_out

    def _calculate_latitude_weights(self):
        '''Calculate latitude-based weights (cos(lat), max=1 at equator)'''
        lats = self.target_grid.lat.values
        lons = self.target_grid.lon.values
        # Calculate cosine of latitude (standard atmospheric weighting)
        lat_weights_1d = np.cos(np.deg2rad(lats))
        lat_weights_2d = np.broadcast_to(
            lat_weights_1d[:, np.newaxis], (len(lats), len(lons)))
        # Create xarray DataArray
        weights_da = xr.DataArray(
            lat_weights_2d,
            coords={'lat': lats, 'lon': lons},
            dims=['lat', 'lon'],
            name='grid_weights',
            attrs={
                'long_name': 'Latitude-based grid weights',
                'description': 'Cosine of latitude weighting',
                'units': 'dimensionless',
                'formula': 'cos(latitude)',
                'note': 'Standard grid weighting for area-weighted means'
            }
        )
        return weights_da

    def _generate_output_filenm(self, dataset_type: str,
                                  target_coll: str = None,
                                  info_dir: str = None,
                                  chunk_id: str = None) -> str:
        '''Generate descriptive output filenm'''
        # Get the appropriate model name
        if dataset_type == 'fcst':
            model_nm = self.fcst_model
        elif dataset_type == 'ana':
            model_nm = self.ana_model
        elif dataset_type == 'clim':
            model_nm = self.clim_model
        else:
            raise ValueError(f'Invalid dataset_type: {dataset_type}. Must be '
                             f'one of: forecast, analysis, climatology')
        prefix = dataset_type

        # Simple name when chunk_id is presnet
        if chunk_id is not None:
            filenm = f'{prefix}_{chunk_id}.nc4'
        # Standard detailed name
        else:
            # Create grid size string using actual dimensions
            grid_size = f'{self.nlat}x{self.nlon}'
            # Get common parameters
            start_date = self.config['start_date']
            end_date = self.config['end_date']
            date_range = f'{start_date}-{end_date}'
            fcst_length = self.config['FDAYS']
            fcst_interval = int(self.config['NFREQ'][:2]) # Hours from HHMMSS
            fcst_spacing = self.config.get('fcst_spacing', 1)
            # Build filenm with optional collection component
            coll_part = (f'{target_coll}_'
                               if target_coll is not None else '')
            # Add exclusion count for fcst if any dates excluded
            exclude_dates = self.config.get('exclude_dates', [])
            excl_part = (f'_excl{len(exclude_dates)}' if exclude_dates
                         and dataset_type == 'fcst' else '')
            filenm = (f'{prefix}_{coll_part}{model_nm}_{date_range}_'
                        f'len{fcst_length}d_int{fcst_interval}h_'
                        f'spc{fcst_spacing}d{excl_part}_{grid_size}.nc4')
        return filenm

    @staticmethod
    def _add_exclusion_metadata(dataset, config):
        '''Add exclusion metadata to dataset'''
        exclude_dates = config.get('exclude_dates', [])
        if exclude_dates:
            dataset.attrs['excluded_dates'] = [str(d) for d in exclude_dates]
            dataset.attrs['num_excluded_dates'] = len(exclude_dates)
            dataset.attrs['exclusion_note'] = (f'{len(exclude_dates)} dates '
                                               f'excluded from processing')

    def _check_for_existing_datasets(self, stats_only_mode: bool = False
                                     ) -> Dict[str, str]:
        '''Check for already-processed datasets in specified directories'''
        print('\n--- Checking for existing datasets ---')
        # Get search directories from config
        search_dirs = self.config['dir_loc']
        # Build expected files dict for enabled datasets only
        expected_files = {}
        if self.fcst_model and self.fcst_model.strip():
            expected_files['fcst'] = (
                self._generate_output_filenm('fcst'), self.fcst_model)
        if self.ana_model and self.ana_model.strip():
            expected_files['ana'] = (
                self._generate_output_filenm('ana'), self.ana_model)
        if self.clim_model and self.clim_model.strip():
            expected_files['clim'] = (
                self._generate_output_filenm('clim'), self.clim_model)
        if not expected_files:
            print('No datasets enabled for processing')
            return {}
        dataset_types = [f'{dtype} ({model})' for dtype, (_, model)
                         in expected_files.items()]
        print(f'Looking for datasets: {", ".join(dataset_types)}')

        # Search for existing files
        found_datasets = {}
        exclude_dates = self.config.get('exclude_dates', [])
        current_excl_count = len(exclude_dates)
        for dataset_type, (filenm, model_nm) in expected_files.items():
            print(f'\n  Looking for {dataset_type}: {filenm}')
            found_file = None
            # Check all directories for this file
            for search_dir in search_dirs:
                # First try exact match
                full_path = os.path.join(search_dir, filenm)
                if os.path.exists(full_path):
                    found_file = full_path
                    print(f'    [FOUND] {full_path}')
                    break
                else:
                    print(f'    [CHECKED] {full_path} - Not found')
                # If forecast and exclusions, try files with fewer exclusions
                if current_excl_count > 0 and dataset_type == 'fcst':
                    for excl_count in range(current_excl_count):
                        if excl_count == 0:
                            # Try without any exclusion suffix
                            alt_filenm = filenm.replace(
                                f'_excl{current_excl_count}', '')
                        else:
                            # Try with fewer exclusions
                            alt_filenm = filenm.replace(
                                f'_excl{current_excl_count}',
                                f'_excl{excl_count}')
                        alt_path = os.path.join(search_dir, alt_filenm)
                        if os.path.exists(alt_path):
                            found_file = alt_path
                            print(f'    [FOUND] {alt_path} (fewer exclusions)')
                            break
                        else:
                            print(f'    [CHECKED] {alt_path} (fewer '
                                  f'exclusions) - Not found')
                    if found_file:
                        break
            if found_file:
                # Validate dates if the found file has exclusions in its name
                needs_validation = ('_excl' in os.path.basename(found_file)
                                    and dataset_type == 'fcst')
                if needs_validation:
                    exclude_dates = self.config.get('exclude_dates', [])
                    spacing = self.config.get('fcst_spacing', 1)
                    expected_dates = self._parse_date_range_with_spacing(
                        self.config['FDATES'], spacing, exclude_dates)
                    if self._validate_dataset_dates(
                            found_file, expected_dates):
                        found_datasets[dataset_type] = found_file
                        print('    [VALIDATED] Date validation passed')
                    else:
                        print('    [REJECTED] Date validation failed')
                else:
                    found_datasets[dataset_type] = found_file
            else:
                if stats_only_mode:
                    print(f'  [NOT FOUND] {dataset_type}: {filenm}')
                    print(f'    Searched in: {", ".join(search_dirs)}')
                else:
                    print(f'  [NOT FOUND] {dataset_type}: Will process from '
                          f'source files')
        return found_datasets

    def _validate_dataset_dates(self, dataset_path: str,
                                expected_dates: List[datetime]) -> bool:
        '''Validate existing dataset for expected dates after exclusions'''
        try:
            with xr.open_dataset(dataset_path, decode_timedelta=True) as ds:
                actual_dates = [pd.to_datetime(d).to_pydatetime()
                                    for d in ds.init_date.values]
                return set(actual_dates) == set(expected_dates)
        except Exception as e:
            print(f'    [WARNING] Could not validate dates in {dataset_path}: '
                  f'{e}')
            return False

    def _select_pressure_levels(self, ds, var_dict):
        '''Select target pressure levels for 3D variables'''
        print(f"      [INFO]: DATASET DIMENSIONS: {ds.dims}")
        if 'level' in ds.dims:
            level_coord = 'level'
        elif 'lev' in ds.dims:
            level_coord = 'lev'
        elif 'pressure' in ds.dims:
            level_coord = 'pressure'
        elif 'pressure_level' in ds.dims:
            level_coord = 'pressure_level'
        else:  # No pressure levels, return as-is (for 2D variables)
            return ds

        if level_coord:
            available_levels = set(ds[level_coord].values)

        # Check if this is Prithvi model-level data
        level_values = ds[level_coord].values
        is_prithvi = all(lv in PRITHVI_PRESSURE_LVLS for lv in level_values)

        if is_prithvi:
            print(f'    Detected Prithvi model levels, mapping to pressure levels...')

            # Map requested pressures to Prithvi levels
            prithvi_levels, pressure_mapping = map_pressure_to_prithvi_levels(
                self.pressure_levels, PRITHVI_PRESSURE_LVLS)

            if prithvi_levels is None:
                return ds

            # Select the Prithvi levels
            ds_subset = ds.sel({level_coord: prithvi_levels})

            # Rename the levels to the requested pressure values
            # Create mapping from Prithvi level index to pressure value
            new_lev_values = [pressure_mapping[lv] for lv in prithvi_levels]
            ds_subset = ds_subset.assign_coords({level_coord: new_lev_values})

            print(f'    Mapped {len(prithvi_levels)} Prithvi levels to pressures: '
                f'{new_lev_values}')

            # Store the mapping for later reference
            if not hasattr(self, 'prithvi_mapping'):
                self.prithvi_mapping = pressure_mapping
        else:
            # Standard pressure level selection
            ds_subset = ds.sel({level_coord: self.pressure_levels})

        if level_coord != 'lev':
            ds_subset = ds_subset.rename({level_coord: 'lev'})

        return ds_subset

    def _get_time_coordinate_name(self, ds):
        '''Get time coordinate name: time or valid_time'''
        if 'time' in ds.dims or 'time' in ds.coords:
            return 'time'
        elif 'valid_time' in ds.dims or 'valid_time' in ds.coords:
            return 'valid_time'
        else:
            return None

    def _get_date_variables(self, init_date: datetime, valid_time: datetime
                            ) -> Dict[str, str]:
        '''Get standardized date variables for template substitution'''
        return {
            'init_YYYY': init_date.strftime('%Y'),
            'init_MM': init_date.strftime('%m'),
            'init_DD': init_date.strftime('%d'),
            'init_HH': init_date.strftime('%H'),
            'init_YYYYMMDD': init_date.strftime('%Y%m%d'),
            'init_YYYYMMDDHH': init_date.strftime('%Y%m%d%H'),
            'valid_YYYY': valid_time.strftime('%Y'),
            'valid_MM': valid_time.strftime('%m'),
            'valid_DD': valid_time.strftime('%d'),
            'valid_HH': valid_time.strftime('%H'),
            'valid_YYYYMMDD': valid_time.strftime('%Y%m%d'),
            'valid_YYYYMMDDHH': valid_time.strftime('%Y%m%d%H')
        }

    def _get_regridder(self, source_ds, grid_type, collections=None):
        '''Retrieve or create regridder for given source dataset'''
        lat_key = source_ds.lat.shape[0] if 'lat' in source_ds.dims else 0
        lon_key = source_ds.lon.shape[0] if 'lon' in source_ds.dims else 0
        regridder_key = f'{grid_type}_{lat_key}x{lon_key}'
        if regridder_key not in self.regridders:
            print(f'    Creating regridder for {grid_type}')
            try:
                # Use conservative_normed for correcy polar handling
                regridder = xe.Regridder(
                    source_ds, self.target_grid, 'conservative_normed',
                    periodic=True)
                self.regridders[regridder_key] = regridder
            except Exception as e:
                print(f'    [ERROR] Regridder creation failed: {e}')
                return None
        return self.regridders[regridder_key]

    def _get_file_by_templates(self, templates: List[str], base_dir: str,
                               date_vars: Dict[str, str], context_info: str,
                               init_date: datetime = None,
                               valid_time: datetime = None) -> tuple:
        '''
        Universal file finder using multiple templates with shift support
        Returns: (file_path, associated_valid_time, searched_paths) or
            ('', None, searched_paths) if not found
        '''
        all_matches = []
        searched_paths = []
        for template in templates:
            try:
                processed_template = template
                # Handle shift patterns like {init_YYYYMMDD,shift=-3h}
                def replace_shift_tag(match):
                    try:
                        full_tag = match.group(1)
                        if ',' not in full_tag:
                            # No shift, return as-is
                            return '{' + full_tag + '}'
                        tag, shift_part = full_tag.split(',', 1)
                        tag = tag.strip()
                        shift_part = shift_part.strip()
                        if not shift_part.startswith('shift='):
                            # Not a shift, return as-is
                            return '{' + full_tag + '}'
                        # Parse shift amount
                        shift_str = shift_part.replace('shift=', '')
                        try:
                            if shift_str.endswith('h'):
                                shift_hours = float(shift_str[:-1])
                            elif shift_str.endswith('d'):
                                shift_hours = float(shift_str[:-1]) * 24
                            else:
                                shift_hours = float(shift_str)  # Assume hours
                        except ValueError:
                            # Invalid shift, return as-is
                            return '{' + full_tag + '}'
                        # Determine which datetime to shift
                        if tag.startswith('init_') and init_date:
                            shifted_dt = init_date + \
                                timedelta(hours=shift_hours)
                        elif tag.startswith('valid_') and valid_time:
                            shifted_dt = valid_time + \
                                timedelta(hours=shift_hours)
                        else:
                            # Cannot shift, return as-is
                            return '{' + full_tag + '}'
                        # Format the shifted datetime (part after init_/valid_)
                        fmt_part = tag.split('_', 1)[1]
                        if fmt_part == 'YYYY':
                            result = shifted_dt.strftime('%Y')
                        elif fmt_part == 'MM':
                            result = shifted_dt.strftime('%m')
                        elif fmt_part == 'DD':
                            result = shifted_dt.strftime('%d')
                        elif fmt_part == 'HH':
                            result = shifted_dt.strftime('%H')
                        elif fmt_part == 'YYYYMMDD':
                            result = shifted_dt.strftime('%Y%m%d')
                        elif fmt_part == 'YYYYMMDDHH':
                            result = shifted_dt.strftime('%Y%m%d%H')
                        else:
                            return '{' + full_tag + '}'  # Unknown format
                        return result
                    except Exception:
                        return match.group(0)  # Return original on error

                # Replace all shift tags first
                processed_template = re.sub(
                    r'\{([^}]+)\}', replace_shift_tag, processed_template)
                # Then handle any remaining regular tags
                processed_template = processed_template.format(**date_vars)
                # Build path and find files
                if processed_template.startswith('/'):
                    path = processed_template
                else:
                    path = os.path.join(base_dir, processed_template)
                searched_paths.append(path)  # Track this path
                matches = glob.glob(path)
                all_matches.extend(matches)
            except Exception as e:
                print(f'  [ERROR] Template processing failed: {e}')
                continue

        if len(all_matches) == 0:
            return '', None, searched_paths
        # Remove duplicates while preserving order
        unique_matches = list(dict.fromkeys(all_matches))
        if len(unique_matches) == 1:
            return unique_matches[0], valid_time, searched_paths
        elif len(unique_matches) > 1:
            # Multiple unique files for same datetime: proceed with first match
            print(f'  [WARNING] Multiple files found for same valid time: '
                  f'{len(unique_matches)} files')
            print(f'  Context: {context_info}')
            print(f'  Using first file: {os.path.basename(unique_matches[0])}')
            return unique_matches[0], valid_time, searched_paths
        return '', None, searched_paths

    def _get_files_for_dataset(self, dataset_type: str,
                                collections: List[str] = None,
                                init_date: datetime = None, fhour: int = None,
                                valid_time: datetime = None, cycle: str = None
                                ) -> tuple:
        '''
        Get files for all collections of a dataset type
        Returns: (coll_files_dict, file_valid_times_dict, searched_paths_dict)
        '''
        # Map dataset types to directory attributes
        base_dir_mapping = {
            'fcst': self.fcst_dir,
            'ana': self.ana_dir,
            'clim': self.clim_dir
        }

        # Map dataset types to template collection keys
        template_mapping = {
            'fcst': 'fcst',
            'ana': 'ana',
            'clim': 'clim'
        }

        base_dir = base_dir_mapping.get(dataset_type, '')
        template_key = template_mapping.get(dataset_type, dataset_type)
        template_colls = self.template_colls.get(template_key, {})

        # Get collection-specific directories
        dir_colls = self.dir_colls.get(dataset_type, {})

        # Determine date variables and valid time
        if dataset_type == 'fcst' and init_date:
            if fhour is not None:
                # Single-lead case
                if valid_time is None:
                    valid_time = init_date + timedelta(hours=fhour)
                context_info = (f'FORECAST: init='
                                f'{init_date.strftime("%Y-%m-%d %H:%M")} '
                                f'fhour={fhour}')
            else:
                # Multi-lead case: use init_date as valid_time for templates
                valid_time = init_date
                context_info = (f'FORECAST: init='
                                f'{init_date.strftime("%Y-%m-%d %H:%M")} '
                                f'(multi-lead)')
            date_vars = self._get_date_variables(init_date, valid_time)
        elif dataset_type == 'ana' and valid_time:
            date_vars = self._get_date_variables(valid_time, valid_time)
            context_info = (f'ANALYSIS: valid_time='
                            f'{valid_time.strftime("%Y-%m-%d %H:%M")}')
        elif dataset_type == 'clim' and cycle:
            dummy_time = datetime(2000, 1, 1, int(cycle))
            valid_time = dummy_time
            date_vars = self._get_date_variables(dummy_time, dummy_time)
            context_info = f'CLIMATOLOGY: cycle={cycle}z'
        else:
            return {}, {}, {}

        # Find files for collections that have templates
        coll_files = {}
        coll_valid_times = {}
        searched_paths = {}  # Store paths that were searched but not found
        used_files = set()
        colls_without_templates = []

        # Use specified collections or find all collections with variables
        if collections is not None:
            needed_colls = collections
        else:
            needed_colls = []
            for coll_nm, vars_dict in self.var_colls.items():
                if vars_dict['3d'] or vars_dict['2d']:
                    needed_colls.append(coll_nm)

        # Loop through all needed collections (specified or have vars defined)
        for coll_nm in needed_colls:
            # Only look for files if this collection has a template
            if (coll_nm in template_colls
                and template_colls[coll_nm]):
                # Collection has specific template
                templates = template_colls[coll_nm]
            elif (coll_nm == 'default'
                and 'default' in template_colls
                and template_colls['default']):
                # Default collection uses default template
                templates = template_colls['default']
            else:
                # No template provided for this collection
                colls_without_templates.append(coll_nm)
                continue

            # Determine the base directory for this collection
            # Priority: collection-specific dir > default dir > base dir
            if coll_nm in dir_colls:
                # Collection has its own directory
                coll_base_dir = os.path.join(base_dir, dir_colls[coll_nm])
            elif 'default' in dir_colls:
                # Use default collection directory
                coll_base_dir = os.path.join(base_dir, dir_colls['default'])
            else:
                # No collection directories specified, use base
                coll_base_dir = base_dir

            file_path, file_valid_time, paths = self._get_file_by_templates(
                templates, coll_base_dir,
                date_vars,
                f'{context_info} collection={coll_nm}',
                init_date, valid_time
            )

            if file_path:
                # Only add if we have not seen this file before
                if file_path not in used_files:
                    coll_files[coll_nm] = file_path
                    coll_valid_times[coll_nm] = file_valid_time
                    used_files.add(file_path)
            else:
                # Store the searched paths for this missing collection
                searched_paths[coll_nm] = paths

        return coll_files, coll_valid_times, searched_paths

    def _parse_date_range_with_spacing(self, fdates_str: str, spacing: int,
                                   exclude_dates: List = None
                                   ) -> List[datetime]:
        '''Parse FDATES string with forecast spacing'''
        start_str, end_str = fdates_str.split('-')
        start_date = datetime.strptime(start_str, '%Y%m%d')
        end_date = datetime.strptime(end_str, '%Y%m%d')
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=spacing)
        # Apply exclusions if provided
        if exclude_dates:
            exclude_datetime_set = set()
            for exclude_date in exclude_dates:
                if isinstance(exclude_date, (int, str)):
                    exclude_datetime_set.add(datetime.strptime(
                        str(exclude_date), '%Y%m%d'))
            dates = [d for d in dates if d not in exclude_datetime_set]
        return dates

    def _generate_leads(self, fdays: int, nfreq: str) -> List[int]:
        '''Generate leads based on FDAYS and NFREQ'''
        freq_hours = int(nfreq[:2])
        max_hours = fdays * 24
        return list(range(0, max_hours + 1, freq_hours))

    def find_match(self, dataset_vars, var, var_aliases, coll_nm,
                   calc_var=None):
        '''
        Helper function to search for direct or alias match.
        Modifies coll_results in place.
        '''
        found_nm = None
        # Direct match
        if var.lower() in dataset_vars:
            found_nm = dataset_vars[var.lower()]
        # Alias match
        elif var in var_aliases:
            for alias in var_aliases[var]:
                if alias in dataset_vars:
                    found_nm = dataset_vars[alias]
                    break
            if not found_nm:
                print(f'[WARNING] {var} not found in collection {coll_nm} '
                      f'(aliases: {var_aliases[var]})')
        else:
            print(f'[WARNING] Missing alias definition for {var} - no '
                  f'{var}_alias in YAML')
        if found_nm:
            if calc_var:
                print(f'      Found dependency {var} in collection '
                      f'{coll_nm} for calculated variable {calc_var}')
            else:
                print(f'      Found {var} in collection {coll_nm}')
        return found_nm

    def _validate_variables_and_levels(self, coll_files: Dict[str, str],
                                       dataset_type: str,
                                       skip_calc_mode: bool = False,
                                       existing_dataset_mode: bool = False
                                       ) -> Dict[str, Any]:
        '''
        Validate that all required variables and pressure levels exist
        without processing. Also handle fallback logic for missing collections.
        Returns: validation results dict
        '''

        # Initialize collection-specific results
        coll_results = {}
        for coll_nm in coll_files.keys():
            coll_results[coll_nm] = {
                'found_vars': {},
                'calculated_vars': {}
            }

        # Get all requested variables across available collections
        available_colls = set(coll_files.keys())
        requested_vars = set()
        requested_3d_vars = set()
        for coll_nm in available_colls:
            if coll_nm in self.var_colls:
                coll_vars = self.var_colls[coll_nm]
                requested_vars.update(coll_vars['3d']
                                      + coll_vars['2d'])
                requested_3d_vars.update(coll_vars['3d'])

        # Group collections by file to avoid duplicate processing
        file_to_colls = {}
        for coll_nm, file_path in coll_files.items():
            if file_path not in file_to_colls:
                file_to_colls[file_path] = []
            file_to_colls[file_path].append(coll_nm)

        # Existing dataset mode: just check if final variables exist
        if existing_dataset_mode:
            print(f'    Validating {len(requested_vars)} variables '
                  f'{sorted(requested_vars)} across {len(file_to_colls)}'
                  f' files...')
            missing_vars = set()
            missing_levels = set()
            # Check each file (should be just one for existing datasets)
            for file_path, colls_for_file in file_to_colls.items():
                try:
                    with xr.open_dataset(file_path, decode_timedelta=True
                                         ) as ds:
                        dataset_vars = {var.lower(): var for var
                                        in ds.data_vars.keys()}
                        print(f'    File: {os.path.basename(file_path)} '
                              f'(collections: '
                              f'{", ".join(sorted(colls_for_file))})')
                        # Check which variables exist
                        found_vars = []
                        coll_nms = ', '.join(sorted(colls_for_file))
                        for var in requested_vars:
                            found_nm = self.find_match(dataset_vars, var,
                                                         self.var_aliases,
                                                         coll_nms)
                            if found_nm:
                                found_vars.append(var)
                            else:
                                missing_vars.add(var)
                        print(f'      Variables found: {sorted(found_vars)}')
                        # Check pressure levels for 3D variables
                        found_3d_vars = [var for var in found_vars if var
                                        in requested_3d_vars]
                        print(f"    Found 3d vars: {found_3d_vars}")
                        if found_3d_vars:
                            level_coord = None
                            for coord in ['level', 'lev', 'pressure']:
                                if coord in ds.dims:
                                    level_coord = coord
                                    break
                            if level_coord:
                                available_levels = set(ds[level_coord].values)
                                print(f"Found available levels: {available_levels}")
                                available_set = set(int(lv) for lv in available_levels)

                                # Check if this is Prithvi model-level data
                                # is_prithvi = all(lv in PRITHVI_PRESSURE_LVLS
                                #                 for lv in available_levels)
                                # Check if these are Prithvi pressure values (not model indices)
                                prithvi_pressure_values = set(PRITHVI_PRESSURE_LVLS.values())
                                is_prithvi = all(lv in prithvi_pressure_values for lv in available_levels)
                                print(f"is_prithvi: {is_prithvi}")

                                if is_prithvi:
                                    print(f"      Detected Prithvi pressure levels (already mapped to standard levels)")
                                    # These are already the standard pressure levels, just validate they exist
                                    found_levels = sorted(available_levels & set(self.pressure_levels))
                                    file_missing_levels = set(self.pressure_levels) - available_levels
                                    if file_missing_levels:
                                        missing_levels.update(file_missing_levels)
                                        print(f'      Pressure levels: {found_levels} '
                                            f'(missing: {sorted(file_missing_levels)})')
                                    else:
                                        print(f'      Pressure levels: {found_levels} (complete)')
                                else:
                                    # Standard pressure level validation
                                    found_levels = sorted(available_levels & set(
                                        self.pressure_levels))
                                    file_missing_levels = set(self.pressure_levels
                                                            ) - available_levels
                                    if file_missing_levels:
                                        missing_levels.update(file_missing_levels)
                                        print(f'      Pressure levels: {found_levels} '
                                            f'(missing: '
                                            f'{sorted(file_missing_levels)})')
                                    else:
                                        print(f'      Pressure levels: {found_levels} '
                                            f'(complete)')
                            else:
                                print('      ERROR: 3D variables found but no '
                                    'pressure coordinate')
                                missing_levels.update(self.pressure_levels)

                except Exception as e:
                    traceback.print_exc()
                    raise ValueError(f'Cannot read {dataset_type} file '
                                     f'{file_path}: {e}')
            # Final check and return
            if missing_vars or missing_levels:
                error_parts = []
                if missing_vars:
                    error_parts.append(f'Variables not found: '
                                       f'{sorted(missing_vars)}')
                if missing_levels:
                    error_parts.append(f'Missing pressure levels: '
                                       f'{sorted(missing_levels)}')
                raise ValueError(f'{dataset_type} validation failed - '
                                 f'{"; ".join(error_parts)}')
            print('    [OK] All required variables and levels present')
            return {'coll_results': {},
                    'files_checked': list(file_to_colls.keys())}

        # Multi-collection mode: initialize tracking for calculable variables
        if not skip_calc_mode:
            requested_calc_vars = {}
            unresolved_calc_vars = set()
            needed_dependencies = set()
            # Find which collections need which calculable variables
            for coll_nm in available_colls:
                if coll_nm in self.var_colls:
                    coll_vars = self.var_colls[coll_nm]
                    needed_vars = coll_vars['3d'] + coll_vars['2d']
                    for calc_var, dependencies in CALCULABLE_VARS.items():
                        if calc_var in needed_vars:
                            if coll_nm not in requested_calc_vars:
                                requested_calc_vars[coll_nm] = []
                            requested_calc_vars[coll_nm].append(calc_var)
                            unresolved_calc_vars.add(calc_var)
                            needed_dependencies.update(dependencies)

        # Collection-specific mode: find all requested calculable variables
        else:
            all_requested_calc_vars = set()
            # Use the original collections before filtering
            for coll_nm in self.original_var_colls:
                coll_vars = self.original_var_colls[coll_nm]
                needed_vars = coll_vars['3d'] + coll_vars['2d']
                for calc_var in CALCULABLE_VARS.keys():
                    if calc_var in needed_vars:
                        all_requested_calc_vars.add(calc_var)

        # Report strategy
        print(f'    Validating {len(requested_vars)} variables '
              f'{sorted(requested_vars)} across {len(file_to_colls)} '
              f'files...')
        if not skip_calc_mode and requested_calc_vars:
            print(f'    Note: Will check if variables can be calculated if not'
                  f' found directly: {list(unresolved_calc_vars)}')
        elif skip_calc_mode and all_requested_calc_vars:
            print(f'    Note: Will preserve dependencies for calculable '
                  f'variables if not found directly: '
                  f'{sorted(all_requested_calc_vars)}')

        # Process each unique file
        missing_vars = set()
        missing_levels = set()
        for file_path, colls_for_file in file_to_colls.items():
            try:
                with xr.open_dataset(file_path, decode_timedelta=True) as ds:
                    dataset_vars = {var.lower(): var
                                    for var in ds.data_vars.keys()}
                    # Collection-specific mode: check while file is open
                    if skip_calc_mode:
                        # Calculate which variables still need dependencies
                        all_found_vars = set()
                        for coll_result in coll_results.values():
                            all_found_vars.update(coll_result[
                                'found_vars'].keys())
                        unfound_calc_vars = (all_requested_calc_vars
                                             - all_found_vars)
                        if unfound_calc_vars:
                            print('    Checking for dependencies of calculable'
                                  'variables...')
                            for coll_nm in colls_for_file:
                                if coll_nm in coll_results:
                                    coll_result = coll_results[coll_nm]
                                    for calc_var in unfound_calc_vars:
                                        dependencies = (
                                            CALCULABLE_VARS.get(
                                                calc_var, []))
                                        for dep_var in dependencies:
                                            if dep_var not in coll_result[
                                                    'found_vars']:
                                                found_nm = self.find_match(
                                                    dataset_vars, dep_var,
                                                    self.var_aliases,
                                                    coll_nm, calc_var)
                                                if found_nm:
                                                    coll_result[
                                                        'found_vars'][
                                                            dep_var
                                                            ] = found_nm
                    # Main validation for this file
                    print(f'    File: {os.path.basename(file_path)} '
                          f'(collections: '
                          f'{", ".join(sorted(colls_for_file))})')
                    # Check which requested variables are in this file
                    for coll_nm in colls_for_file:
                        coll_vars = self.var_colls.get(
                            coll_nm, {'3d': [], '2d': []})
                        needed_vars = (coll_vars['3d']
                                       + coll_vars['2d'])
                        # Make sure collection is in results
                        if coll_nm not in coll_results:
                            coll_results[coll_nm] = {
                                'found_vars': {},
                                'calculated_vars': {}
                            }
                        coll_found_vars = coll_results[coll_nm][
                            'found_vars']
                        # Process all requested variables
                        for var in needed_vars:
                            if var in coll_found_vars:
                                continue
                            found_nm = self.find_match(dataset_vars, var,
                                                         self.var_aliases,
                                                         coll_nm)
                            if found_nm:
                                coll_found_vars[var] = found_nm
                            # Mark calculable variable as resolved
                            # (multi-collection mode only)
                                if (not skip_calc_mode
                                    and var in unresolved_calc_vars):
                                    unresolved_calc_vars.discard(var)
                        # Multi-collection mode: Check for dependencies of
                        # unresolved calculable variables
                        if not skip_calc_mode:
                            for dep_var in needed_dependencies:
                                # Check if any unresolved calculable variable
                                # still needs this dependency
                                dependency_still_needed = False
                                for calc_var in unresolved_calc_vars:
                                    if (calc_var in CALCULABLE_VARS
                                        and dep_var
                                        in CALCULABLE_VARS[calc_var]):
                                        dependency_still_needed = True
                                        break
                                if not dependency_still_needed:
                                    continue
                                # Only process if not already found somewhere
                                already_found = any(dep_var in other_results[
                                    'found_vars'] for other_results in
                                    coll_results.values())
                                if (not already_found and dep_var
                                    not in coll_found_vars):
                                    found_nm = self.find_match(
                                        dataset_vars, dep_var,
                                        self.var_aliases, coll_nm)
                                    if found_nm:
                                        coll_found_vars[dep_var] = found_nm
                    # Report variables found in this file
                    file_found_vars = set()
                    for coll_nm in colls_for_file:
                        if coll_nm in coll_results:
                            file_found_vars.update(coll_results[
                                coll_nm]['found_vars'].keys())
                    print(f'      Variables found: {sorted(file_found_vars)}')

                    # Check pressure levels for 3D variables
                    file_3d_vars = [var for var in file_found_vars
                                    if var in requested_3d_vars]
                    if file_3d_vars:
                        level_coord = None
                        for coord in ['level', 'lev', 'pressure']:
                            if coord in ds.dims:
                                level_coord = coord
                                break
                        if level_coord:
                            available_levels = list(ds[level_coord].values)

                            # Check if this is Prithvi model-level data
                            is_prithvi = all(int(lv) in PRITHVI_PRESSURE_LVLS for lv in available_levels)

                            if is_prithvi:
                                # Map requested pressures to Prithvi levels
                                prithvi_levels, pressure_mapping = map_pressure_to_prithvi_levels(
                                    self.pressure_levels, PRITHVI_PRESSURE_LVLS)

                                if prithvi_levels:
                                    # Check if all needed Prithvi levels exist
                                    available_set = set(int(lv) for lv in available_levels)
                                    missing_prithvi = [lv for lv in prithvi_levels
                                                    if lv not in available_set]
                                    if missing_prithvi:
                                        missing_pressures = [PRITHVI_PRESSURE_LVLS[lv]
                                                            for lv in missing_prithvi]
                                        missing_levels.update(missing_pressures)
                                        found_pressures = [pressure_mapping[lv]  # Use requested pressures
                                                        for lv in prithvi_levels
                                                        if lv in available_set]
                                        print(f'      Prithvi pressure levels: '
                                            f'{sorted(found_pressures)} '
                                            f'(missing: {sorted(missing_pressures)})')
                                    else:
                                        found_pressures = [pressure_mapping[lv]  # Use requested pressures
                                                        for lv in prithvi_levels]
                                        print(f'      Prithvi pressure levels: '
                                            f'{sorted(found_pressures)} (complete)')
                            else:
                                # Standard pressure level validation
                                available_levels_set = set(available_levels)
                                found_levels = sorted(available_levels_set & set(self.pressure_levels))
                                file_missing_levels = set(self.pressure_levels) - available_levels_set
                                if file_missing_levels:
                                    missing_levels.update(file_missing_levels)
                                    print(f'      Pressure levels: {found_levels} '
                                        f'(missing: {sorted(file_missing_levels)})')
                                else:
                                    print(f'      Pressure levels: {found_levels} (complete)')
            except Exception as e:
                raise ValueError(f'Cannot read {dataset_type} file '
                                 f'{file_path}: {e}')

        # Multi-collection mode:
        # Check if remaining calculable variables can be resolved
        if not skip_calc_mode and unresolved_calc_vars:
            print('    Checking if remaining variables can be calculated...')

            for coll_nm, calc_vars in requested_calc_vars.items():
                for calc_var in calc_vars:
                    if calc_var in unresolved_calc_vars:
                        dependencies = CALCULABLE_VARS[calc_var]
                        accumulated_deps = {}
                        # Look for each dependency across all collections
                        for dep_var in dependencies:
                            for other_coll, other_result in (
                                    coll_results.items()):
                                if dep_var in other_result['found_vars']:
                                    accumulated_deps[dep_var] = {
                                        'collection': other_coll,
                                        'alias': other_result[
                                            'found_vars'][dep_var]
                                    }
                                    break
                        # Check if all dependencies were found
                        if len(accumulated_deps) == len(dependencies):
                            coll_results[coll_nm][
                                'calculated_vars'][calc_var
                                                        ] = accumulated_deps
                            coll_results[coll_nm][
                                'found_vars'][calc_var] = 'calculated'
                            deps_info = ', '.join([
                                f'{dep} from {info["collection"]}'
                                for dep, info in accumulated_deps.items()])
                            print(f'      {calc_var} can be calculated using '
                                  f'{deps_info}')
                        else:
                            missing_deps = [dep for dep in dependencies
                                            if dep not in accumulated_deps]
                            print(f'      {calc_var} cannot be calculated - '
                                  f'missing: {missing_deps}')

        # Final validation
        all_found_vars = set()
        for coll_result in coll_results.values():
            all_found_vars.update(coll_result['found_vars'].keys())
        missing_vars = requested_vars - all_found_vars

        # Collection-specific mode: no error for missing calculable variables
        if skip_calc_mode:
            calculable_vars = set(CALCULABLE_VARS.keys())
            missing_vars = missing_vars - calculable_vars
        # Return error for missing variables or levels and return result
        if missing_vars or missing_levels:
            error_parts = []
            if missing_vars:
                error_parts.append(f'Variables could not be found or '
                                   f'calculated: {sorted(missing_vars)}')
            if missing_levels:
                error_parts.append(f'Missing pressure levels: '
                                   f'{sorted(missing_levels)}')
            raise ValueError(f'{dataset_type} validation failed - '
                             f'{"; ".join(error_parts)}')
        return {
            'coll_results': coll_results,
            'files_checked': list(set(coll_files.values()))
        }

    def _validate_dataset(self, dataset_type, all_files, no_template_colls,
                      skip_calc_mode=False, coll_modes=None, leads=None):
        '''Validate with sample files'''

        try:
            # Get needed collections
            needed_colls = []
            for coll_nm, vars_dict in self.var_colls.items():
                if vars_dict['3d'] or vars_dict['2d']:
                    needed_colls.append(coll_nm)
            # Get sample files for each collection
            sample_files = {}
            for coll_nm in needed_colls:
                # Find a sample file for this collection
                sample_file = None
                for key, files in all_files[dataset_type].items():
                    if coll_nm in files:
                        sample_file = files[coll_nm]
                        break
                if sample_file:
                    sample_files[coll_nm] = sample_file
            # Find which collections need template fallback
            collections_needing_fallback = []
            for coll, needed in no_template_colls.items():
                if needed and coll not in sample_files:
                    collections_needing_fallback.append(coll)
            # Look for any variables which need fallback in default files
            if collections_needing_fallback:
                if 'default' not in sample_files:
                    raise ValueError(
                        f'Collections {collections_needing_fallback} need '
                        f'default template fallback, but default template '
                        f'is missing for {dataset_type} dataset')
                for coll in collections_needing_fallback:
                    # Update ALL keys in all_files for this dataset type
                    for key in all_files[dataset_type].keys():
                        if 'default' in all_files[dataset_type][key]:
                            all_files[dataset_type][key][coll] = all_files[
                                dataset_type][key]['default']
                    # Also add to sample_files for validation
                    sample_files[coll] = sample_files['default']

            # Validate with sample_files
            validation_result = self._validate_variables_and_levels(
                sample_files, dataset_type, skip_calc_mode)
            # Additional validation for multi-lead forecast collections
            if dataset_type == 'fcst' and coll_modes:
                for coll_nm, file_path in sample_files.items():
                    if (coll_nm in coll_modes
                        and coll_modes[coll_nm] == 'multi_lead'):
                        print(f'      Checking lead times for multi-lead '
                              f'collection {coll_nm}...')
                        try:
                            with xr.open_dataset(file_path,
                                                 decode_timedelta=True) as ds:
                                if 'time' in ds.coords:
                                    # Get datetime objects
                                    time_values = ds.time.values
                                    first_time = time_values[0]
                                    # Calculate lead hours as integers
                                    available_leads = []
                                    for time_val in time_values:
                                        # Convert timedelta to hours as integer
                                        lead_hours = int(
                                            (pd.to_datetime(time_val)
                                             - pd.to_datetime(first_time)
                                             ).total_seconds() / 3600)
                                        available_leads.append(lead_hours)
                                    # Check if all required leads are present
                                    missing_leads = []
                                    for required_lead in leads:
                                        if (required_lead
                                            not in available_leads):
                                            missing_leads.append(required_lead)
                                    if missing_leads:
                                        return (f'FCST validation failed for '
                                                f'collection {coll_nm}: '
                                                f'Missing required lead times:'
                                                f' {missing_leads}. '
                                                f'(Available leads: '
                                                f'{sorted(available_leads)})')
                                    else:
                                        available_count = len(available_leads)
                                        required_count = len(leads)
                                        print(f'        Lead times: '
                                              f'{required_count}/'
                                              f'{available_count} required '
                                              f'leads found')
                                else:
                                    return (f'FCST validation failed for '
                                            f'collection {coll_nm}: No time '
                                            f'coordinate found in multi-lead '
                                            f'file')
                        except Exception as e:
                            traceback.print_exc()
                            return (f'FCST validation failed for collection '
                                    f'{coll_nm}: Error reading time '
                                    f'coordinates: {e}')
            print(f'    [OK] {dataset_type.capitalize()} validation passed - '
                  'all required variables and pressure levels found')
            return validation_result
        except Exception as e:
            return f'{dataset_type.capitalize()} validation failed: {e}'

    def _calculate_monthly_weights(self, dt: datetime) -> tuple:
        '''Calculate distance-based weights for monthly interpolation'''

        day_with_hours = dt.day - 1 + dt.hour / 24.0
        month = dt.month
        year = dt.year

        # Get days in current month
        days_in_month = calendar.monthrange(year, month)[1]
        month_center = days_in_month / 2

        if day_with_hours < month_center:
            # Closer to start of month: blend with previous month
            prev_month = month - 1 if month > 1 else 12
            prev_year = year if month > 1 else year - 1

            # Get days in previous month for its center
            days_in_prev_month = calendar.monthrange(prev_year, prev_month)[1]
            prev_month_center = days_in_prev_month / 2

            # Calculate distances to both centers
            dist_to_curr = month_center - day_with_hours
            dist_to_prev = day_with_hours + prev_month_center
            total_dist = dist_to_curr + dist_to_prev

            # Weight is inversely proportional to distance
            weight_prev = dist_to_curr / total_dist
            weight_curr = dist_to_prev / total_dist
            return (prev_month, prev_year, weight_prev), (month, year,
                                                          weight_curr)

        elif day_with_hours > month_center:
            # Closer to end of month: blend with next month
            next_month = month + 1 if month < 12 else 1
            next_year = year if month < 12 else year + 1

            # Get days in next month for its center
            days_in_next_month = calendar.monthrange(next_year, next_month)[1]
            next_month_center = days_in_next_month / 2

            # Calculate distances to both centers
            dist_to_curr = day_with_hours - month_center
            dist_to_next = next_month_center + (days_in_month - day_with_hours)
            total_dist = dist_to_curr + dist_to_next

            # Weight is inversely proportional to distance
            weight_curr = dist_to_next / total_dist
            weight_next = dist_to_curr / total_dist
            return (month, year, weight_curr), (next_month, next_year,
                                                weight_next)

        else:
            # Exactly at center: return same month twice with 50/50 weights
            return (month, year, 0.5), (month, year, 0.5)

    def _process_dataset_variables(self, ds, colls_for_file, file_type,
                                   validation_result):
        '''
        Find and regrid variables from a dataset (using pre-validated results)
        Returns:
        - regridded_vars: dict of regridded variables {var_nms: dataarray}
        - processed_collections: list of successfully processed collections
        '''

        # Handle different time coordinate names by normalizing to 'time'
        time_coord_nm = self._get_time_coordinate_name(ds)
        if time_coord_nm and time_coord_nm != 'time':
            ds = ds.rename({time_coord_nm: 'time'})

        # Extract found_vars and calculated_vars from validation results
        found_vars = {}
        calculated_vars = {}
        for coll_nm in colls_for_file:
            if coll_nm in validation_result['coll_results']:
                coll_result = validation_result['coll_results'][coll_nm]
                found_vars.update(coll_result['found_vars'])
                calculated_vars.update(coll_result['calculated_vars'])
        if not found_vars:
            # No variables to process for these collections
            return {}, colls_for_file
        # Select pressure levels only if needed
        pressure_coord_names = ['lev', 'level', 'pressure', 'pressure_level']
        needs_pressure_levels = any(
            any(coord in ds[var].dims for coord in pressure_coord_names)
            for var in found_vars.values() if var != 'calculated'
        )
        if needs_pressure_levels:
            ds_levels = self._select_pressure_levels(ds, found_vars)
        else:
            ds_levels = ds

        # Get regridder
        regridder = self._get_regridder(ds_levels, file_type,
                                        colls_for_file)
        if regridder is None:
            print(f'[ERROR] Failed to create regridder for {file_type}')
            return {}, []

        # Process variables, including dependencies for calculated variables
        regridded_vars = {}
        # Define standardized coords (for cross-collection diffs and shifts)
        if file_type == 'clim':
            std_coords = np.arange(1, 13, dtype=np.int32)  # 1-12 for months
        else:
            std_coords = np.array([0], dtype=np.int32)  # 0 for single time
        for target_var, source_var in found_vars.items():
            if source_var == 'calculated':
                # Extract and process dependencies for this calculated variable
                if target_var in calculated_vars:
                    for dep_var, dep_info in calculated_vars[
                            target_var].items():
                        # Only process dependencies for collections in file
                        if (dep_info['collection'] in colls_for_file
                            and dep_var not in regridded_vars):
                            dep_source = dep_info['alias']
                            try:
                                regridded_data = regridder(
                                    ds_levels[dep_source])
                                if 'time' in regridded_data.coords:
                                    regridded_data = (
                                        regridded_data.assign_coords(
                                            time=std_coords))
                                regridded_vars[dep_var] = regridded_data
                                print(f'  [INFO] Processed dependency '
                                      f'{dep_var} (alias: {dep_source}) for '
                                      f'calculated variable {target_var}')
                            except Exception as e:
                                # Propagate the error up
                                raise ValueError(f'Failed to process '
                                                 f'dependency {dep_var} '
                                                 f'({dep_source}) for '
                                                 f'calculated variable '
                                                 f'{target_var}: {str(e)}')
                continue  # Skip the calculated variable itself
            try:
                regridded_data = regridder(ds_levels[source_var])
                if 'time' in regridded_data.coords:
                    regridded_data = (
                        regridded_data.assign_coords(time=std_coords))
                regridded_vars[target_var] = regridded_data
            except Exception as e:
                print(f'[ERROR] Failed to regrid {target_var} ({source_var}): '
                      f'{str(e)}')

        return regridded_vars, colls_for_file

    def _regrid_multi_lead_datasets(self, multi_lead_datasets, needed_fhours,
                                    file_type, validation_result,
                                    coll_modes, no_template_colls):
        '''
        Regrid multi-lead datasets once for all needed fhours
        Returns: regridded_cache: Dict {coll_nm: {fhour: {var_nms: dataarray}}}
        '''
        regridded_cache = {}
        for coll_nm, ds in multi_lead_datasets.items():
            coll_nms = [coll_nm]
            # Add template-less collections if processing default
            if coll_nm == 'default':
                if no_template_colls:
                    for key in no_template_colls.keys():
                        coll_nms.append(key)
            for coll_nm in coll_nms:
                regridded_cache[coll_nm] = {}
            # Get datetime objects
            time_values = ds.time.values
            first_time = time_values[0]  # Reference time (lead = 0)
            # Calculate lead hours as integers and create mapping to indices
            time_indices = {}
            for i, time_val in enumerate(time_values):
                # Convert timedelta to hours as integer
                lead_hours = int((pd.to_datetime(time_val) -
                                  pd.to_datetime(first_time)).total_seconds()
                                 / 3600)
                time_indices[lead_hours] = i
            # Process each needed fhour
            for fhour in needed_fhours:
                # Extract this fhour
                time_idx = time_indices[fhour]
                fhour_ds = ds.isel(time=time_idx, drop=True)
                # Process variables for this fhour
                try:
                    regridded_vars, _ = self._process_dataset_variables(
                        fhour_ds, coll_nms, file_type, validation_result)
                except Exception as e:
                    raise ValueError(f'[Failed to regrid variables for '
                                     f'{coll_nms} collection at fhour='
                                     f'{fhour}: {str(e)}')
                for coll_nm in coll_nms:
                    regridded_cache[coll_nm][fhour] = regridded_vars

        return regridded_cache

    def _combine_results(self, single_result, multi_regridded_vars, file_type,
                         collections_processed):
        '''
        Combine single-lead results with multi-lead regridded variables
        Parameters:
        - single_result: xr.Dataset from single-lead processing (or None)
        - multi_regridded_vars: Dict {var_nms: dataarray} from multi-lead
        - file_type: str
        - collections_processed: List of collection names
        Returns:
        - Combined xr.Dataset or None
        '''

        all_vars = {}
        # Add single-lead variables
        if single_result is not None:
            for var in single_result.data_vars:
                all_vars[var] = single_result[var]
        # Add multi-lead variables
        if multi_regridded_vars:
            all_vars.update(multi_regridded_vars)
        if not all_vars:
            return None
        return xr.Dataset(all_vars)

    def _load_and_process_files(self, coll_files: Dict[str, str],
                                file_type: str, validation_result=None):
        '''Load and process files from multiple collections'''

        # Group collections by file path to avoid duplicate processing
        file_to_colls = {}
        for coll_nm, file_path in coll_files.items():
            if file_path not in file_to_colls:
                file_to_colls[file_path] = []
            file_to_colls[file_path].append(coll_nm)

        # Process each file
        all_regridded_vars = {}
        for file_path, colls_for_file in file_to_colls.items():
            try:
                ds = xr.open_dataset(file_path, decode_timedelta=True)

                regridded_vars, _ = self._process_dataset_variables(
                    ds, colls_for_file, file_type, validation_result)

                all_regridded_vars.update(regridded_vars)
                ds.close()
            except Exception as e:
                traceback.print_exc()
                raise ValueError(f'File processing failed for {file_path}: '
                                 f'{e}')
        return xr.Dataset(all_regridded_vars)

    def _load_and_process_files_for_init_date(
            self, files_by_key, file_type, coll_modes, validation_result,
            no_template_colls):
        '''
        Process multiple keys for a single init_date (multi-lead file handling)
        Parameters:
        files_by_key: dict mapping keys (init_date, fhour) to collection files
        file_type: str, type of file being processed
        coll_modes: dict mapping collections to single_lead or multi_lead
        Returns: list[xr.Dataset]: processed results, one per fhour (in order)
        '''

        # Separate multi-lead and single-lead templated collections
        multi_lead_colls  = [c for c, mode in coll_modes.items()
                                   if mode == 'multi_lead']
        single_lead_colls = [c for c, mode in coll_modes.items()
                                   if mode == 'single_lead']
        # Add template-less collections while preserving templated multi list
        multi_lead_templated = multi_lead_colls.copy()
        if no_template_colls:
            default_mode = coll_modes['default']
            for coll_nm in no_template_colls.keys():
                if default_mode == 'multi_lead':
                    multi_lead_colls.append(coll_nm)
                else:
                    single_lead_colls.append(coll_nm)

        # Sort keys by fhour
        sorted_keys = sorted(files_by_key.keys(), key=lambda k: k[1])
        init_date = sorted_keys[0][0]
        needed_fhours = [k[1] for k in sorted_keys]
        print(f'  Processing init_date={init_date.strftime("%Y-%m-%d %H:%M")} '
              f'for {len(sorted_keys)} leads')

        # Process by key
        results = []
        regridded_multi_cache = {}  # Cache for regridded multi-lead data
        for i, key in enumerate(sorted_keys):
            fhour = key[1]
            coll_files = files_by_key[key]
            # Separate files by type
            multi_files  = {name: path for name, path in
                            coll_files.items() if name in
                            multi_lead_templated}
            if i==0:
            # First fhour: process multi-lead files
                # Open and cache multi-lead datasets
                multi_lead_datasets = {}
                if multi_files:
                    for coll_nm, file_path in multi_files.items():
                        try:
                            ds = xr.open_dataset(file_path,
                                                 decode_timedelta=True)
                            # Make sure it is actually a multi-lead file
                            if 'time' in ds.dims and ds.time.size > 1:
                                multi_lead_datasets[coll_nm] = ds
                            else:
                                # Not actually multi-lead
                                ds.close()
                                raise ValueError(f'File for collection '
                                                 f'{coll_nm} was marked as '
                                                 f'multi-lead but does not '
                                                 f'have multiple time points.')
                        except Exception as e:
                            traceback.print_exc()
                            raise ValueError(f'Multi-lead file processing '
                                             f'failed for {file_path}: {e}')

                # Regrid multi-lead datasets for all fhours
                if multi_lead_datasets:
                    try:
                        regridded_multi_cache = (
                            self._regrid_multi_lead_datasets(
                                multi_lead_datasets, needed_fhours, file_type,
                                validation_result, coll_modes,
                                no_template_colls))
                        # Close originals now that we have the regridded cache
                        for ds in multi_lead_datasets.values():
                            ds.close()
                    except Exception as e:
                        for ds in multi_lead_datasets.values():
                            ds.close()
                        raise ValueError(f'Multi-lead regridding failed: {e}')

            # Process single-lead files
            single_files = {name: path for name, path in
                            coll_files.items() if name in
                            single_lead_colls}
            single_result = None
            if single_files:
                single_result = self._load_and_process_files(
                    single_files, file_type, validation_result)
            # Get cached multi-lead data for this fhour
            multi_vars_for_fhour = {}
            collections_used = list(
                single_files.keys()) if single_files else []
            for coll_nm in multi_lead_colls:
                if (coll_nm in regridded_multi_cache and
                    fhour in regridded_multi_cache[coll_nm]):
                    multi_vars_for_fhour.update(regridded_multi_cache[
                        coll_nm][fhour])
                    collections_used.append(coll_nm)
            # Combine results
            combined_result = self._combine_results(
                single_result, multi_vars_for_fhour, file_type,
                collections_used)
            results.append(combined_result)

        return results

    def _process_files_sequential(self, all_coll_files: Dict[
            str, Dict[str, str]], file_type: str, validation_result=None,
            coll_modes=None, no_template_colls=None) -> Dict:
        '''Process files sequentially'''

        # Inititalize calculation of file numbers
        total_files = len(all_coll_files)
        file_to_colls = {}
        coll_file_counts = {}
        processed_data = {}
        failed_count = 0
        for key, coll_files in all_coll_files.items():
            for coll_nm, file_path in coll_files.items():
                if file_path not in file_to_colls:
                    file_to_colls[file_path] = set()
                file_to_colls[file_path].add(coll_nm)
                if coll_nm not in coll_file_counts:
                    coll_file_counts[coll_nm] = 0
                coll_file_counts[coll_nm] += 1
        unique_files = len(file_to_colls)

        # Check if any multi-lead collections for forecast files
        has_multi_lead = False
        if file_type == 'fcst' and coll_modes:
            multi_lead_colls  = [c for c, mode in
                                       coll_modes.items()
                                       if mode == 'multi_lead']
            single_lead_colls = [c for c, mode in
                                       coll_modes.items()
                                       if mode == 'single_lead']
            has_multi_lead = bool(multi_lead_colls)
            if has_multi_lead:
                print(f'  Found {len(multi_lead_colls)} multi-lead '
                      f'collections: {multi_lead_colls}')
                if single_lead_colls:
                    print(f'  Found {len(single_lead_colls)} single-lead'
                          f' collections: {single_lead_colls}')
                    print('  Using optimized processing')

        # If forecast with multi-lead collections, use optimized processing
        if file_type == 'fcst' and has_multi_lead:
            # Group keys by init_date
            keys_by_init_date = {}
            for key in all_coll_files.keys():
                init_date = key[0]  # Extract init_date from key
                if init_date not in keys_by_init_date:
                    keys_by_init_date[init_date] = []
                keys_by_init_date[init_date].append(key)
            # Process keys by init_date
            init_counter = 0
            total_inits = len(keys_by_init_date)
            processed_keys_count = 0  # Initialize counter for processed keys
            for init_date, keys_for_init in sorted(keys_by_init_date.items()):
                # Collect files for this init_date
                keys_for_init.sort(key=lambda k: k[1])  # Sort by fhour
                files_for_init = {k: all_coll_files[k]
                                  for k in keys_for_init}
                # Process all keys for this init_date together
                init_results = self._load_and_process_files_for_init_date(
                    files_for_init, file_type, coll_modes,
                    validation_result, no_template_colls)
                init_counter += 1
                # Process results and update key count
                successful_keys = 0
                if init_results and len(init_results) == len(keys_for_init):
                    key_result_pairs = [(keys_for_init[i], init_results[i])
                                        for i in range(len(keys_for_init))]
                    for key, result in key_result_pairs:
                        if result is not None:
                            processed_data[key] = result
                            successful_keys += 1
                        else:
                            failed_count += 1
                else:
                    # All files failed or mismatch in results
                    failed_count += len(keys_for_init)
                # Update count and print progress every other init date
                processed_keys_count += successful_keys
                if init_counter % 2 == 0 or init_counter == total_inits:
                    print(f'  [PROGRESS] Processed {init_counter}/'
                          f'{total_inits} init dates')
                # Convert list results back to dict format
                if init_results and len(init_results) == len(keys_for_init):
                    key_result_pairs = [(keys_for_init[i], init_results[i])
                                        for i in range(len(keys_for_init))]
                    for key, result in key_result_pairs:
                        if result is not None:
                            processed_data[key] = result
                        else:
                            failed_count += 1
                else:
                    # All files failed or mismatch in results
                    failed_count += len(keys_for_init)
        else:
            # Standard processing for ana/clim or forecast without multi-lead
            file_counter = 0
            # Process by keys
            for key, coll_files in all_coll_files.items():
                file_counter += 1
                result = self._load_and_process_files(
                    coll_files, file_type, validation_result)
                # Print progress every 5 files
                if file_counter % 5 == 0 or file_counter == total_files:
                    print(f'  [PROGRESS] Processed {file_counter}/'
                          f'{total_files} time points')
                if result is not None:
                    processed_data[key] = result
                else:
                    failed_count += 1

        # Calculate and print summary
        success_count = len(processed_data)
        total_count = total_files
        if failed_count > 0:
            print(f'[ERROR] Processing failed: {success_count}/{total_count} '
                  f'time points processed ({failed_count} failed)')
            raise ValueError(f'{file_type} processing failed: {failed_count} '
                             f'out of {total_count} time points could not be '
                             f'processed')
        else:
            print(f'[OK] Processed: {unique_files} files across {total_count} '
                  f'time points (all successful)')

        return processed_data

    def _combine_into_structured_datasets(
            self, loaded_data, init_dates, init_dates_full, leads,
            val_results=None, skip_calc_mode=False):
        '''Combine individual files into structured datasets (clim/ana/fcst)'''

        print('\n--- Combining into structured datasets ---')
        combined_datasets = {}

        # CLIMATOLOGY: process/combine for valid times
        if 'clim' in loaded_data and loaded_data['clim']:
            print('Building climatology dataset...')
            # Generate all possible valid times from init dates and leads
            valid_times = []
            for init_date in init_dates_full:
                for fhour in leads:
                    valid_time = init_date + timedelta(hours=fhour)
                    valid_times.append(valid_time)
            # Remove duplicates and sort
            unique_valid_times = sorted(list(set(valid_times)))
            print(f'    Generated {len(unique_valid_times)} unique valid times'
                  f' for climatology')
            start_time = unique_valid_times[0]
            end_time = unique_valid_times[-1]

            # Calculate weights for first and last time to show as examples
            first_weights = self._calculate_monthly_weights(start_time)
            last_weights = self._calculate_monthly_weights(end_time)
            first_month1, first_year1, first_weight1 = first_weights[0]
            first_month2, first_year2, first_weight2 = first_weights[1]
            last_month1,  last_year1,  last_weight1  = last_weights[0]
            last_month2,  last_year2,  last_weight2  = last_weights[1]

            # Show summary of climatology processing (with examples)
            print( '    Creating time-interpolated climatologies')
            print(f'    Time range: {start_time} to {end_time}')
            print( '    Monthly weighting examples:')
            print(f'      {start_time} = {first_weight1:.1%} month '
                  f'{first_month1} + {first_weight2:.1%} month {first_month2}')
            print(f'      {end_time} = {last_weight1:.1%} month '
                  f'{last_month1} + {last_weight2:.1%} month {last_month2}')

            # Process climatology for each time point
            clim_list = []
            times = []
            missing_cycles = []
            for time_val in unique_valid_times:
                # Get the appropriate cycle data for this time
                cycle = f'{time_val.hour:02d}'
                if cycle in loaded_data['clim']:
                    ds = loaded_data['clim'][cycle]

                    # Check if time dimension exists and has expected size
                    if 'time' in ds.dims and ds.time.size != 12:
                        raise ValueError(f'Climatology data for cycle '
                                         f'{cycle}z has {ds.time.size} time '
                                         f'steps (expected 12 months)')
                    elif 'time' not in ds.dims:
                        raise ValueError(f'Climatology data for cycle '
                                         f'{cycle}z missing time dimension '
                                         f'(expected 12 months)')

                    # Calculate monthly weights for this datetime
                    (month1, year1, weight1), (
                        month2, year2, weight2
                        ) = self._calculate_monthly_weights(time_val)
                    # Apply monthly interpolation (weighted average)
                    if 'time' in ds.dims and ds.time.size == 12:
                        # Get the two months (convert to 0-based indexing)
                        month1_data = ds.isel(time=month1-1)
                        month2_data = ds.isel(time=month2-1)
                        interpolated_ds = (weight1 * month1_data
                                           + weight2 * month2_data)
                        # Remove the time dimension
                        if 'time' in interpolated_ds.dims:
                            interpolated_ds = interpolated_ds.squeeze(
                                'time', drop=True)
                    else:
                        # No monthly dimension: use as-is
                        interpolated_ds = ds
                        if 'time' in interpolated_ds.dims:
                            interpolated_ds = interpolated_ds.squeeze(
                                'time', drop=True)
                    times.append(time_val)
                    clim_list.append(interpolated_ds)

                else:
                    missing_cycles.append((cycle, time_val))

            # ERROR CHECK: Ensure we have climatology data
            if not clim_list:
                raise ValueError('No climatology data found for any required '
                                 'cycles')

            # ERROR CHECK: Report missing cycles
            if missing_cycles:
                missing_info = [f'{cycle}z' for cycle, _ in missing_cycles]
                raise ValueError(f'Missing climatology data for cycles: '
                                 f'{missing_info}')

            # Combine climatology
            for i, ds in enumerate(clim_list):
                clim_list[i] = ds.assign_coords(time=[times[i]])
            combined_clim = xr.concat(clim_list, dim='time')
            combined_clim = self._clean_and_filter_dataset(
                combined_clim, ['time', 'lev', 'lat', 'lon'], 'clim',
                val_results, skip_calc_mode)
            combined_datasets['clim'] = combined_clim
            print(f'  [OK] Climatology combined: {dict(combined_clim.sizes)}')
            print(f'    Variables: {list(combined_clim.data_vars.keys())}')

        # ANALYSIS: process/combine for valid times
        if 'ana' in loaded_data and loaded_data['ana']:
            print('Building analysis dataset...')
            ana_list = []
            times = []
            for time_key, ds in loaded_data['ana'].items():
                # Parse time key format (robust for different models)
                try:
                    time_val = pd.to_datetime(time_key)
                except:
                    try:
                        formatted_time_key = time_key.replace(
                            '_', ' ') + ':00:00'
                        time_val = pd.to_datetime(formatted_time_key)
                    except:
                        try:
                            iso_format = time_key.replace('_', 'T') + ':00:00'
                            time_val = pd.to_datetime(iso_format)
                        except:
                            try:
                                if '_' in time_key:
                                    date_part, hour_part = time_key.split('_')
                                    time_val = datetime.strptime(
                                        f'{date_part}T{hour_part}:00:00',
                                        '%Y-%m-%dT%H:%M:%S')
                                else:
                                    raise ValueError('Unknown format')
                            except Exception as e:
                                print(f'    Warning: Could not parse time key:'
                                      f' {time_key} ({e})')
                                continue
                times.append(time_val)
                ana_list.append(ds)

            # Sort by time and combine
            sorted_pairs = sorted(zip(times, ana_list))
            sorted_ana = [ds for _, ds in sorted_pairs]
            sorted_times = [t for t, _ in sorted_pairs]
            for i, ds in enumerate(sorted_ana):
                sorted_ana[i] = ds.assign_coords(time=[sorted_times[i]])
            combined_ana = xr.concat(sorted_ana, dim='time')
            combined_ana = self._clean_and_filter_dataset(
                combined_ana, ['time', 'lev', 'lat', 'lon'], 'ana',
                val_results, skip_calc_mode)
            combined_datasets['ana'] = combined_ana
            print(f'  [OK] Analysis: {dict(combined_ana.sizes)}')
            print(f'    Variables: {list(combined_ana.data_vars.keys())}')

        # FORECAST: process/combine by init_date and lead
        if 'fcst' in loaded_data and loaded_data['fcst']:
            print('Building forecast dataset...')
            forecast_dict = {}
            for (init_date, fhour), ds in loaded_data['fcst'].items():
                # Remove any existing time dimension
                if 'time' in ds.dims:
                    if ds.time.size == 1:
                        ds = ds.squeeze('time', drop=True)
                    else:
                        ds = ds.isel(time=0, drop=True)

                if init_date not in forecast_dict:
                    forecast_dict[init_date] = {}

                ds_with_lead = ds.assign_coords(lead=fhour)
                forecast_dict[init_date][fhour] = ds_with_lead

            # Combine into structured dataset
            init_date_datasets = []
            for init_date in sorted(forecast_dict.keys()):
                lead_datasets = []
                for fhour in sorted(forecast_dict[init_date].keys()):
                    lead_datasets.append(forecast_dict[init_date][fhour])

                init_ds = xr.concat(lead_datasets, dim='lead')
                init_ds = init_ds.assign_coords(init_date=init_date)
                init_date_datasets.append(init_ds)
            combined_fcst = xr.concat(init_date_datasets, dim='init_date')
            combined_fcst = self._clean_and_filter_dataset(
                combined_fcst, ['init_date', 'lead', 'lev', 'lat', 'lon'],
                'fcst', val_results, skip_calc_mode)
            combined_fcst['lead'].attrs[
                'long_name'] = 'Forecast lead time (hours)'
            combined_datasets['fcst'] = combined_fcst
            print(f'  [OK] Forecast: {dict(combined_fcst.sizes)}')
            print(f'    Variables: {list(combined_fcst.data_vars.keys())}')

        # Add grid weights to all
        for name, ds in combined_datasets.items():
            ds['grid_weights'] = self.grid_weights
        print('\nFinal dataset coordinates:')
        for name, ds in combined_datasets.items():
            print(f'  {name}: {list(ds.coords.keys())}')

        return combined_datasets

    def _calculate_variables(self, ds, calculated_vars):
        '''Calculate any calculable variables and add them to dataset'''
        for calc_var, dependencies in calculated_vars.items():
            # Check if all dependencies are available
            missing_deps = []
            for dep_var in dependencies.keys():
                if dep_var not in ds.data_vars:
                    missing_deps.append(dep_var)
            if missing_deps:
                raise ValueError(f'Cannot calculate {calc_var}: missing '
                                 f'required dependencies {missing_deps}')
            print(f'  Calculating {calc_var} from {list(dependencies.keys())}')

            try:
                if calc_var == 'D2m':
                    # Add units to the xarray DataArrays
                    q2m_with_units = ds['Q2m'] * units('kg/kg')
                    ps_with_units = ds['PS'] * units.Pa
                    # Calculate dewpoint using metpy (returns in Celsius)
                    dewpoint_celsius = mpcalc.dewpoint_from_specific_humidity(
                        ps_with_units, q2m_with_units)
                    celsius_values = dewpoint_celsius.values
                    kelvin_values = celsius_values + 273.15
                    # Add D2m to the dataset with proper dimensions
                    ds[calc_var] = xr.DataArray(
                        kelvin_values,
                        coords=dewpoint_celsius.coords,
                        dims=dewpoint_celsius.dims)
                elif calc_var == 'LOGAOD':
                    ds[calc_var] = np.log(ds['AOD']+.01)
                    ds[calc_var].attrs={
                        'calculation_method': (
                            f'Calculated from '
                            f'{", ".join(list(dependencies.keys()))}')}
                elif calc_var == 'Z':  # ADD THIS BLOCK
                    # Calculate geopotential from geopotential height
                    # Z = g * H, where g = 9.80665 m/sÂ²
                    g = 9.80665  # standard gravity in m/sÂ²
                    ds[calc_var] = ds['H'] * g
                    ds[calc_var].attrs = {
                        'calculation_method': (
                            f'Calculated from H using Z = g * H '
                            f'(g = {g} m/sÂ²)'),
                        'long_name': 'Geopotential',
                        'units': 'mÂ²/sâ»Â²'
                    }
                elif calc_var == 'PM25':
                    ds[calc_var] = (ds['SSSMASS25']  + ds['DUSMASS25']
                                    + ds['BCSMASS']  + ds['OCSMASS']
                                    + ds['SO4SMASS'] + ds['NISMASS25']
                                    + ds['NH4SMASS'])
                    ds[calc_var].attrs={
                        'calculation_method': (
                            f'Calculated from '
                            f'{", ".join(list(dependencies.keys()))}')}

                print(f'  [SUCCESS] {calc_var} calculation complete')
            except Exception as e:
                raise ValueError(f'Failed to calculate {calc_var}: '
                                 f'{str(e)}')
        return ds

    def _clean_and_filter_dataset(self, dataset, essential_coords,
                                  dataset_type=None, val_results=None,
                                  skip_calc_mode=False):
        '''
        Clean and filter a dataset to keep only essential vars and coords
        Parameters:
        dataset: xarray.Dataset to clean and filter
        essential_coords: list of coordinate names to keep
        dataset_type: type of dataset ('fcst', 'ana', 'clim') [optional]
        val_results: dict of validation results with calculated vars info
        Returns:
        xarray.Dataset with only essential variables and coordinates
        '''

        # Calculate derived variables if validation results provided
        if dataset_type and val_results and dataset_type in val_results:
            calculated_vars = {}
            dependencies = set()

            for coll_result in val_results[dataset_type]['coll_results'
                                                         ].values():
                if 'calculated_vars' in coll_result:
                    if skip_calc_mode:
                        # Extract dependencies for preservation
                        for calc_var, deps in coll_result['calculated_vars'
                                                          ].items():
                            for dep_var in deps.keys():
                                dependencies.add(dep_var)
                    else:
                        calculated_vars.update(coll_result['calculated_vars'])
            if calculated_vars and not skip_calc_mode:
                print(f'  [INFO] Calculating derived variables: '
                      f'{list(calculated_vars.keys())}')
                dataset = self._calculate_variables(dataset, calculated_vars)
            elif skip_calc_mode and dependencies:
                print(f'  [INFO] Preserving dependencies for later merge: '
                      f'{sorted(dependencies)}')

        # Define essential variables from our configuration
        essential_vars = list(self.all_vars_3d) + list(self.all_vars_2d
                                                       ) + ['grid_weights']
        # Collection-specific mode: add extra found variables to essential
        if (skip_calc_mode and val_results
            and dataset_type in val_results):
            for coll_result in val_results[dataset_type]['coll_results'
                                                         ].values():
                for found_var in coll_result['found_vars'].keys():
                    if found_var not in essential_vars:
                        essential_vars.append(found_var)
                        print(f'  [INFO] Preserving found variable: '
                              f'{found_var}')

        # Clean up each data variable to be kept
        for var_nms in list(dataset.data_vars):
            if var_nms in essential_vars:
                # Remove coordinates
                if 'coordinates' in dataset[var_nms].attrs:
                    del dataset[var_nms].attrs['coordinates']
        # Drop everything not in essential list
        vars_to_drop = [var for var in dataset.variables if var
                        not in essential_vars and var not in essential_coords]
        if vars_to_drop:
            print(f'  [INFO] Removing non-essential variables/coordinates: '
                  f'{vars_to_drop}')
            dataset = dataset.drop_vars(vars_to_drop)
        # Add regrid_method as global attribute (removing from individual vars)
        dataset.attrs['regrid_method'] = 'conservative_normed'
        for var_nms in dataset.data_vars:
            if 'regrid_method' in dataset[var_nms].attrs:
                del dataset[var_nms].attrs['regrid_method']

        return dataset

    def _inspect_combined_data(self, combined_datasets):
        '''Inspect the final combined datasets'''
        print('\n--- Final Combined Datasets ---')

        for name, ds in combined_datasets.items():
            print(f'\n{name.upper()} DATASET:')
            print(f'  Dimensions: {dict(ds.sizes)}')
            print(f'  Variables: {list(ds.data_vars.keys())}')
            print(f'  Coordinates: {list(ds.coords.keys())}')

            # Show coordinate ranges
            for coord in ds.coords:
                if coord in ['time', 'init_date', 'lead', 'cycle']:
                    coord_vals = ds.coords[coord].values
                    if len(coord_vals) > 0:
                        if coord in ['time', 'init_date']:
                            print(f'    {coord}: '
                                  f'{pd.to_datetime(coord_vals[0])} to '
                                  f'{pd.to_datetime(coord_vals[-1])}')
                        else:
                            print(f'    {coord}: {coord_vals[0]} to '
                                  f'{coord_vals[-1]}')

    def process_batch(self, target_coll=None, info_dir=None,
                      date_start_idx=None, date_end_idx=None, check_only=False,
                      skip_calc_mode=False, single_fcst_mode=None):
        '''Main batch processing method'''
        if target_coll:
            print(f'\n=== Collection-Specific Batch Processing: '
                  f'{target_coll} ===')
        else:
            print('\n=== Multi-Variable Batch Forecast Processing ===')
        print(f'Target variables: 3D={self.all_vars_3d}, '
              f'2D={self.all_vars_2d}')

        if single_fcst_mode:
            # Create init_dates from single date
            single_date = datetime.strptime(single_fcst_mode, '%Y%m%d')
            init_dates = [single_date]
            init_dates_full = [single_date]
            print(f'Single forecast mode: processing date '
                  f'{single_date.strftime("%Y-%m-%d")}')
        else:
            # Parse date range from config accounting for spacing/exclusions
            spacing = self.config.get('fcst_spacing', 1)
            exclude_dates = self.config.get('exclude_dates', [])
            # Create full date range
            init_dates_full = self._parse_date_range_with_spacing(
                self.config['FDATES'], spacing, [])  # No exclusions
            # Create exclusion-filtered date range
            init_dates = self._parse_date_range_with_spacing(
                self.config['FDATES'], spacing, exclude_dates)
        leads = self._generate_leads(
            self.config['FDAYS'], self.config['NFREQ'])

        # Filter dates by indices if specified (chunks for parallel processing)
        if date_start_idx is not None or date_end_idx is not None:
            # Get valid indices for the full date range
            start_idx = 0 if date_start_idx is None else max(0, min(
                date_start_idx, len(init_dates_full)-1))
            end_idx = len(init_dates_full)-1 if date_end_idx is None else max(
                0, min(date_end_idx, len(init_dates_full)-1))
            # Apply chunking to full date range first
            if start_idx <= end_idx:
                init_dates_full = init_dates_full[start_idx:end_idx+1]
                # Then apply exclusions to the chunked range
                if exclude_dates:
                    exclude_datetime_set = set()
                    for exclude_date in exclude_dates:
                        if isinstance(exclude_date, (int, str)):
                            exclude_datetime_set.add(datetime.strptime(
                                str(exclude_date), '%Y%m%d'))
                    init_dates = [d for d in init_dates_full
                                  if d not in exclude_datetime_set]
                else:
                    init_dates = init_dates_full.copy()

                print(f'[INFO] Processing chunk: indices {start_idx}-'
                      f'{end_idx}, {len(init_dates_full)} dates, '
                      f'{len(init_dates)} after exclusions')

                # Graceful exit if all forecast dates in chunk were excluded
                if len(init_dates) == 0:
                    print('[INFO] All init_dates in this chunk were excluded')
                    print('[SUCCESS] No processing needed for this chunk')
                    return {'status': 'success', 'reason': 'no_dates_in_chunk'}
            else:
                print(f'[ERROR] Invalid date range: start_idx {start_idx} > '
                      f'end_idx {end_idx}')
                return {'status': 'failed',
                        'reason': (f'Invalid date range: start_idx {start_idx}'
                                   f' > end_idx {end_idx}')}
        if single_fcst_mode:
            print(f'Processing single forecast date: '
                  f'{init_dates[0].strftime("%Y-%m-%d")}')
        else:
            print(f'Processing dates: '
                  f'{[d.strftime("%Y-%m-%d") for d in init_dates]} (spacing: '
                  f'{spacing} days)')
        print(f'Leads: {leads}')

        # Check which datasets to process from YAML config and args
        process_fcst = bool(self.fcst_model and self.fcst_model.strip())
        process_ana  = bool(self.ana_model  and self.ana_model.strip())
        process_clim = bool(self.clim_model and self.clim_model.strip())

        # Show dataset processing status with names
        dataset_status = []
        if process_fcst:
            dataset_status.append(f'Forecast={self.fcst_model}')
        if process_ana:
            dataset_status.append(f'Analysis={self.ana_model}')
        if process_clim:
            dataset_status.append(f'Climatology={self.clim_model}')
        print(f'Dataset processing: {", ".join(dataset_status)}')

        # Check for and validate existing datasets
        if single_fcst_mode is None:
            existing_datasets = self._check_for_existing_datasets()
        else:
            existing_datasets = {}
            print('[INFO] Single forecast mode: skipping existing dataset '
                  'check')
        if existing_datasets:
            print('\n--- Validating existing datasets ---')
        validation_errors = []
        datasets_to_reprocess = []
        for dataset_type, file_path in list(existing_datasets.items()):
            if ((dataset_type == 'fcst' and not process_fcst) or
                (dataset_type == 'ana' and not process_ana) or
                (dataset_type == 'clim' and not process_clim)):
                continue
            print(f'\n  Validating {dataset_type}: '
                  f'{os.path.basename(file_path)}')
            try:
                # Load dataset for validation
                with xr.open_dataset(file_path, decode_timedelta=True) as ds:
                    # Establishe all collections for validation
                    mock_coll_files = {}
                    needed_colls = []
                    for coll_nm, vars_dict in self.var_colls.items():
                        if vars_dict['3d'] or vars_dict['2d']:
                            needed_colls.append(coll_nm)
                    for collection in needed_colls:
                        mock_coll_files[collection] = file_path
                    # Validate with the same method used for new files
                    self._validate_variables_and_levels(
                        mock_coll_files, dataset_type,
                        existing_dataset_mode=True)

                    # Successfully validated - update processing flag
                    if dataset_type == 'fcst':
                        process_fcst = False
                        print('    [INFO] Skipping forecast processing - '
                              'using validated dataset')
                    elif dataset_type == 'ana':
                        process_ana = False
                        print('    [INFO] Skipping analysis processing - '
                              'using validated dataset')
                    elif dataset_type == 'clim':
                        process_clim = False
                        print('    [INFO] Skipping climatology processing - '
                              'using validated dataset')

            except Exception as e:
                error_msg = str(e)
                validation_errors.append(f'{dataset_type.capitalize()} '
                                         f'validation failed: {error_msg}')
                print(f'    [ERROR] Validation failed: {error_msg}')
                # Mark for reprocessing
                datasets_to_reprocess.append(dataset_type)

        # Remove items that failed validation
        for dataset_type in datasets_to_reprocess:
            del existing_datasets[dataset_type]
            if dataset_type == 'fcst':
                process_fcst = True
                print('\n[INFO] Forecast dataset failed validation - '
                      'will be reprocessed')
            elif dataset_type == 'ana':
                process_ana = True
                print('\n[INFO] Analysis dataset failed validation - '
                      'will be reprocessed')
            elif dataset_type == 'clim':
                process_clim = True
                print('\n[INFO] Climatology dataset failed validation - '
                      'will be reprocessed')

        # Print which datasets need processing
        datasets_to_process = []
        if process_fcst:
            datasets_to_process.append('fcst')
        if process_ana:
            datasets_to_process.append('ana')
        if process_clim:
            datasets_to_process.append('clim')
        if datasets_to_process:
            print(f'\n[INFO] Will process these datasets: '
                  f'{", ".join(datasets_to_process)}')

        # If check_only mode, write status file and exit
        if check_only:
            # Create job directory and status output file
            job_dir = f'outputs/{info_dir}/jobs'
            os.makedirs(job_dir, exist_ok=True)
            status_file = f'{job_dir}/dataset_status.txt'
            # Write status file
            need_fcst = 'true' if process_fcst else 'false'
            need_ana  = 'true' if process_ana  else 'false'
            need_clim = 'true' if process_clim else 'false'
            with open(status_file, 'w') as f:
                f.write(f'fcst_needed:{need_fcst}\n')
                f.write(f'ana_needed:{need_ana}\n')
                f.write(f'clim_needed:{need_clim}\n')
            # Print summary
            print(f'[CHECK_ONLY] Dataset status written to: {status_file}')
            print('[SUCCESS] Dataset check completed')
            return {
                'status': 'check_only_success',
                'status_file': status_file,
                'datasets_needed': datasets_to_process
            }

        # If all datasets are validated, return success
        if not any([process_fcst, process_ana, process_clim]):
            print('\n[SUCCESS] All datasets validated successfully')
            return {
                'status': 'success',
                'used_existing': True,
                'existing_datasets': existing_datasets,
                'dataset_files': {
                    'fcst': existing_datasets.get('fcst'),
                    'ana': existing_datasets.get('ana'),
                    'clim': existing_datasets.get('clim')
                },
                'init_dates': init_dates,
                'leads': leads
            }

        # Collection-specific mode: filter for specified collection
        if target_coll:
            print(f'\nFiltering for specific collection: {target_coll}')
            # Filter var_colls and all_vars list to the requested collection
            if target_coll in self.var_colls:
                filtered_collections = {
                    target_coll: self.var_colls[target_coll]}
                self.var_colls = filtered_collections
                self.all_vars_3d, self.all_vars_2d = self._get_all_variables()
                print(f'Collection variables: '
                      f'3D={self.var_colls[target_coll]["3d"]}, '
                      f'2D={self.var_colls[target_coll]["2d"]}')
            else:
                available_colls = list(self.var_colls.keys())
                raise ValueError(f'Requested collection {target_coll} not '
                                 f'found. Available: {available_colls}')


        # Step 1: Check complete file availability
        print('\n--- Step 1: Checking complete file availability ---')
        all_files =     {'fcst': {}, 'ana': {}, 'clim': {}}
        missing_files = {'fcst': [], 'ana': [], 'clim': []}
        missing_paths = {'fcst': {}, 'ana': {}, 'clim': {}}

        # Track collections without templates
        no_template_fcst_colls = {}
        no_template_ana_colls = {}
        no_template_clim_colls = {}

        # Detect forecast template modes (single-lead or multi-lead)
        coll_modes = {}
        if process_fcst:
            print('  Detecting forecast template modes...')
            fcst_template_colls = self.template_colls.get('fcst', {})
            for coll_nm in self.var_colls.keys():
                # Skip collections that are empty (no requested variables)
                if not (self.var_colls[coll_nm]['3d']
                        or self.var_colls[coll_nm]['2d']):
                    continue
                if (coll_nm in fcst_template_colls
                    and fcst_template_colls[coll_nm]):
                    templates = fcst_template_colls[coll_nm]
                    # Check if any template contains valid_* variables
                    has_valid_vars = any(
                        'valid_' in template for template in templates)
                    # Validate consistency within collection
                    valid_counts = sum(1 for template in templates
                                       if 'valid_' in template)
                    if 0 < valid_counts < len(templates):
                        raise ValueError(
                            f'Inconsistent template modes in forecast '
                            f'collection {coll_nm}: some templates contain '
                            f'valid_* variables (single-lead) while others do '
                            f'not (multi-lead). All templates in a collection '
                            f'use the same mode.'
                        )
                    coll_modes[coll_nm] = (
                        'single_lead' if has_valid_vars else 'multi_lead')
                    print(f'    Collection {coll_nm}: '
                          f'{coll_modes[coll_nm]}')
            # Track collections without templates for forecasts
            needed_fcst_colls = []
            for coll_nm, vars_dict in self.var_colls.items():
                if vars_dict['3d'] or vars_dict['2d']:
                    needed_fcst_colls.append(coll_nm)
            # Track which collections don't have templates
            for coll_nm in needed_fcst_colls:
                if coll_nm not in coll_modes:  # Not in coll_modes: no template
                    no_template_fcst_colls[coll_nm] = True

        # Get unique hours needed for climatology (from leads)
        unique_hours = set()
        for fhour in leads:
            hour = fhour % 24
            unique_hours.add(f'{hour:02d}')
        needed_cycles = sorted(unique_hours)

        # Calculate unique valid times for analysis files (full date range)
        unique_valid_times = set()
        for init_date in init_dates_full:
            for fhour in leads:
                valid_time = init_date + timedelta(hours=fhour)
                unique_valid_times.add(valid_time)
        unique_valid_times = sorted(list(unique_valid_times))

        # Count files to check (only for enabled datasets)
        total_files_to_check = 0
        if process_fcst:
            # Count based on template modes
            fcst_files_count = 0
            for coll_nm in coll_modes.keys():
                # single-lead: one file per (init, lead)
                if coll_modes[coll_nm] == 'single_lead':
                    fcst_files_count += len(init_dates) * len(leads)
                # multi-lead: one file per init_date
                else:
                    fcst_files_count += len(init_dates)
            total_files_to_check += fcst_files_count
        if process_ana:
            total_files_to_check += len(unique_valid_times)
        if process_clim:
            total_files_to_check += len(needed_cycles)

        # Display file count details
        file_set_details = []
        if process_fcst:
            file_set_details.append(f'forecast: {len(init_dates)} dates Ã— '
                                    f'{len(leads)} leads = '
                                    f'{len(init_dates) * len(leads)}')
        if process_ana:
            file_set_details.append(
                f'analysis: {len(unique_valid_times)} unique valid times')
        if process_clim:
            file_set_details.append(
                f'climatology: {len(needed_cycles)} cycles')
        print(f'  Checking availability of {total_files_to_check} total file '
              f'sets...')
        for detail in file_set_details:
            print(f'    {detail}')

        # Check forecast files (by collection mode)
        single_lead_colls = [c for c, mode in coll_modes.items()
                             if mode == 'single_lead']
        multi_lead_colls  = [c for c, mode in coll_modes.items()
                             if mode == 'multi_lead']
        # Process single-lead collections
        if single_lead_colls:
            for init_date in init_dates:
                for fhour in leads:
                    key = (init_date, fhour)
                    fcst_files, fcst_valid_times, fcst_paths = (
                        self._get_files_for_dataset(
                            'fcst', collections=single_lead_colls,
                            init_date=init_date, fhour=fhour))
                    missing_collections = [c for c in single_lead_colls
                                           if c not in fcst_files]
                    if missing_collections:
                        # Handle missing files
                        missing_key = (
                            f'init={init_date.strftime("%Y-%m-%d %H:%M")} '
                            f'fhour={fhour}')
                        for coll in missing_collections:
                            missing_files['fcst'].append(
                                f'{missing_key} (collection: {coll})')
                        # Store paths
                        if missing_key not in missing_paths['fcst']:
                            missing_paths['fcst'][missing_key] = {}
                        for coll in missing_collections:
                            missing_paths['fcst'][missing_key][
                                coll] = fcst_paths.get(
                                    coll, ['No search paths recorded'])
                    else:
                        # Store found collections for this (init_date, fhour)
                        all_files['fcst'][key] = fcst_files
        # Process multi-lead collections
        if multi_lead_colls:
            for init_date in init_dates:
                fcst_files, fcst_valid_times, fcst_paths = (
                    self._get_files_for_dataset(
                        'fcst', collections=multi_lead_colls,
                        init_date=init_date))
                missing_collections = [c for c in multi_lead_colls
                                       if c not in fcst_files]
                if missing_collections:
                    # Handle missing files
                    missing_key = (
                        f'init={init_date.strftime("%Y-%m-%d %H:%M")}')
                    for coll in missing_collections:
                        missing_files['fcst'].append(
                            f'{missing_key} (collection: {coll})')
                    # Store paths
                    if missing_key not in missing_paths['fcst']:
                        missing_paths['fcst'][missing_key] = {}
                    for coll in missing_collections:
                        missing_paths['fcst'][missing_key][
                            coll] = fcst_paths.get(
                                coll, ['No search paths recorded'])
                else:
                    # Store after replicating to all fhours
                    for fhour in leads:
                        key = (init_date, fhour)
                        if key not in all_files['fcst']:
                            all_files['fcst'][key] = {}
                        all_files['fcst'][key].update(fcst_files)

        # Check analysis files
        if process_ana:
            needed_ana_collections = []
            for coll_nm, vars_dict in self.var_colls.items():
                if vars_dict['3d'] or vars_dict['2d']:
                    needed_ana_collections.append(coll_nm)
            for valid_time in unique_valid_times:
                ana_files, ana_valid_times, ana_paths = (
                    self._get_files_for_dataset('ana', valid_time=valid_time))
                # Check if all collections with templates were found
                missing_collections = []
                for coll in needed_ana_collections:
                    if coll not in ana_files:
                        # Check if this collection has a template
                        template_key = 'ana'
                        template_colls = self.template_colls.get(template_key,
                                                                 {})
                        if coll in template_colls and template_colls[coll]:
                            missing_collections.append(coll)
                        else:
                            if coll not in no_template_ana_colls:
                                no_template_ana_colls[coll] = True
                if missing_collections:
                    # Handle missing files
                    missing_key = (
                        f'valid_time={valid_time.strftime("%Y-%m-%d %H:%M")}')
                    for coll in missing_collections:
                        missing_files['ana'].append(
                            f'{missing_key} (collection: {coll})')
                    if missing_key not in missing_paths['ana']:
                        missing_paths['ana'][missing_key] = {}
                    for coll in missing_collections:
                        missing_paths['ana'][missing_key][
                            coll] = ana_paths.get(
                                coll, ['No search paths recorded'])
                else:
                    all_files['ana'][valid_time.strftime('%Y-%m-%d_%H')
                                     ] = ana_files

        # Check climatology files
        if process_clim:
            needed_clim_collections = []
            for coll_nm, vars_dict in self.var_colls.items():
                if vars_dict['3d'] or vars_dict['2d']:
                    needed_clim_collections.append(coll_nm)
            for cycle in needed_cycles:
                clim_files, clim_valid_times, clim_paths = (
                    self._get_files_for_dataset('clim', cycle=cycle))
                missing_collections = []
                for coll in needed_clim_collections:
                    if coll not in clim_files:
                        template_key = 'clim'
                        template_colls = self.template_colls.get(template_key,
                                                                 {})
                        if coll in template_colls and template_colls[coll]:
                            missing_collections.append(coll)
                        else:
                            if coll not in no_template_clim_colls:
                                no_template_clim_colls[coll] = True
                if missing_collections:
                    # Handle missing files
                    missing_key = f'cycle={cycle}z'
                    for coll in missing_collections:
                        missing_files['clim'].append(
                            f'{missing_key} (collection: {coll})')
                    if missing_key not in missing_paths['clim']:
                        missing_paths['clim'][missing_key] = {}
                    for coll in missing_collections:
                        missing_paths['clim'][missing_key][
                            coll] = clim_paths.get(
                                coll, ['No search paths recorded'])
                else:
                    all_files['clim'][cycle] = clim_files

        # Check for missing files and exit if any found
        total_missing = 0
        for dataset_type, missing_list in missing_files.items():
            if (dataset_type == 'fcst' and process_fcst or
                dataset_type == 'ana' and process_ana or
                    dataset_type == 'clim' and process_clim):
                total_missing += len(missing_list)
        if total_missing > 0:
            print('\n[ERROR] MISSING FILES DETECTED - STOPPING EXECUTION')
            print(f'Total missing files: {total_missing}')
            for file_type, missing_list in missing_files.items():
                # Only show errors for datasets being processed
                if ((file_type == 'fcst' and process_fcst) or
                    (file_type == 'ana' and process_ana) or
                    (file_type == 'clim' and process_clim)) and missing_list:
                    print(f'\n{file_type.upper()} FILES MISSING ('
                          f'{len(missing_list)}):')
                    for missing in missing_list[:10]:
                        print(f'  [ERROR] {missing}')
                        # Extract the base key for path lookup
                        if '(collection:' in missing:
                            base_key = missing.split(' (collection:')[0]
                            collection = missing.split(
                                '(collection: ')[1].rstrip(')')
                        else:
                            base_key = missing
                            collection = None
                        # Print the searched paths for this missing file
                        if base_key in missing_paths[file_type]:
                            file_missing_paths = missing_paths[file_type
                                                               ][base_key]
                            if collection and collection in file_missing_paths:
                                # Show paths for specific collection
                                paths = file_missing_paths[collection]
                                print(f'    Collection: {collection}')
                                if not paths:
                                    print('      No search paths recorded')
                                elif paths[0].startswith('No'):
                                    print(f'      {paths[0]}')
                                else:
                                    for path in paths:
                                        print(f'      Expected: {path}')
                            else:
                                # Show all collections for this time point
                                for coll, paths in file_missing_paths.items():
                                    print(f'    Collection: {coll}')
                                    if not paths:
                                        print('      No search paths recorded')
                                    elif paths[0].startswith('No'):
                                        print(f'      {paths[0]}')
                                    else:
                                        for path in paths:
                                            print(f'      Expected: {path}')
                    if len(missing_list) > 10:
                        print(f'  ... and {len(missing_list) - 10} more')

            return {
                'status': 'failed',
                'reason': 'missing_files',
                'missing_files': missing_files,
                'total_missing': total_missing
            }

        # Show detailed file summary - only for datasets being processed
        print('\n--- File Summary ---')
        for file_type, file_dict in all_files.items():
            if ((file_type == 'fcst' and process_fcst) or
                (file_type == 'ana' and process_ana) or
                    (file_type == 'clim' and process_clim)) and file_dict:
                # Get all files organized by collection
                files_by_coll = {}
                all_unique_files = set()
                for coll_files in file_dict.values():
                    for coll_nm, file_path in coll_files.items():
                        if coll_nm not in files_by_coll:
                            files_by_coll[coll_nm] = set()
                        files_by_coll[coll_nm].add(file_path)
                        all_unique_files.add(file_path)
                print(f'  {file_type.capitalize()}: {len(all_unique_files)} '
                      f'unique files across {len(files_by_coll)} collections')
                # Show files by collection
                for coll_nm in sorted(files_by_coll.keys()):
                    coll_files_set = files_by_coll[coll_nm]
                    print(f'    Collection {coll_nm}: {len(coll_files_set)} '
                          f'files')
                    for file_path in sorted(coll_files_set):
                        print(f'      - {os.path.basename(file_path)}')

        # Step 2: Validate variables and pressure levels with sample files
        print('\n--- Step 2: Validating variables and pressure levels with '
              'sample files ---')
        # Store validation results
        validation_errors = []
        val_results = {}

        # Validate forecast variables and levels
        if process_fcst:
            print('\n  FORECAST VALIDATION:')
            result = self._validate_dataset(
                'fcst', all_files, no_template_fcst_colls, skip_calc_mode,
                coll_modes, leads)
            if isinstance(result, str):  # error message
                validation_errors.append(result)
            else:
                val_results['fcst'] = result
        # Validate analysis variables and levels
        if process_ana:
            print('\n  ANALYSIS VALIDATION:')
            result = self._validate_dataset(
                'ana', all_files, no_template_ana_colls, skip_calc_mode)
            if isinstance(result, str):  # error message
                validation_errors.append(result)
            else:
                val_results['ana'] = result
        # Validate climatology variables and levels
        if process_clim:
            print('\n  CLIMATOLOGY VALIDATION:')
            result = self._validate_dataset(
                'clim', all_files, no_template_clim_colls, skip_calc_mode)
            if isinstance(result, str):  # error message
                validation_errors.append(result)
            else:
                val_results['clim'] = result

        # Stop if validation failed
        if validation_errors:
            print('\n[ERROR] DATASET VALIDATION FAILED')
            for error in validation_errors:
                print(f'  {error}')
            return {'status': 'failed', 'reason': 'validation failed',
                    'errors': validation_errors}

        print('\n  [SUCCESS] All variables and levels validated successfully')

        # Step 3: Process all files
        print('\n--- Step 3: Processing all files ---')
        loaded_data = {}

        if process_fcst and all_files['fcst']:
            loaded_data['fcst'] = self._process_files_sequential(
                all_files['fcst'], 'fcst', val_results.get('fcst'),
                coll_modes, no_template_fcst_colls)
        if process_ana and all_files['ana']:
            loaded_data['ana'] = self._process_files_sequential(
                all_files['ana'], 'ana', val_results.get('ana'))
        if process_clim and all_files['clim']:
            loaded_data['clim'] = self._process_files_sequential(
                all_files['clim'], 'clim', val_results.get('clim'))

        # Step 4: Combine into structured datasets
        combined_datasets = self._combine_into_structured_datasets(
            loaded_data, init_dates, init_dates_full, leads, val_results,
            skip_calc_mode)
        # Add any existing datasets that had passed validation
        for dataset_type, file_path in existing_datasets.items():
            if dataset_type not in combined_datasets:
                ds = xr.open_dataset(file_path, decode_timedelta=True)
                combined_datasets[dataset_type] = ds
                print(f'  [OK] Added existing {dataset_type} dataset: '
                      f'{dict(ds.sizes)}')

        # Step 5: Final inspection and summary
        self._inspect_combined_data(combined_datasets)
        total_processed = sum(len(data_dict)
                              for data_dict in loaded_data.values())
        total_existing = len(existing_datasets)
        print(f'[OK] Total entries processed: {total_processed}')
        print(f'[OK] Created {len(combined_datasets)} structured datasets '
              f'({total_existing} existing + '
              f'{len(combined_datasets)-total_existing} new)')
        # Create dataset files dictionary for stats processing
        dataset_files = {}
        for dataset_type in ['fcst', 'ana', 'clim']:
            if dataset_type in combined_datasets:
                dataset_files[dataset_type] = self._generate_output_filenm(
                    dataset_type)
        # Add existing datasets to the files dict
        for dataset_type, file_path in existing_datasets.items():
            dataset_files[dataset_type] = file_path
        return {
            'status': 'success',
            'datasets': combined_datasets,
            'dataset_files': dataset_files,
            'existing_datasets': existing_datasets,
            'init_dates': init_dates,
            'leads': leads,
            'val_results': val_results
        }

    def merge_forecast_chunks(self, info_dir: str,
                              save_for_coll_merge: bool = False) -> bool:
        '''
        Merge forecast chunk files with proper naming and save location
        Parameters:
        info_dir: directory containing the tmp folder with chunk files
        Returns: True if merge was successful, False otherwise
        '''
        print('\n==================================================')
        print('MERGING FORECAST CHUNKS')
        print('==================================================')

        # Find and organize chunk files
        pattern = f'outputs/{info_dir}/tmp/fcst_chunk_*.nc4'
        print(f'[INFO] Looking for forecast chunks matching: {pattern}')
        chunk_files = sorted(glob.glob(pattern))
        if not chunk_files:
            print('[ERROR] No forecast chunk files found')
            return False
        print(f'[INFO] Found {len(chunk_files)} forecast chunk files:')
        for file in chunk_files:
            print(f'  - {os.path.basename(file)}')

        try:
            # Generate output filename
            if save_for_coll_merge:
            # Will be collection-merged later: save as fcst_default_* to tmp
                output_file = self._generate_output_filenm(
                    'fcst', target_coll='default')
                output_file = f'outputs/{info_dir}/tmp/{output_file}'
                print('[INFO] Saving to tmp for later collection merging')
            else:
                # Final output: save as fcst_* to output directory
                output_file = self._generate_output_filenm('fcst')
                output_file = f'outputs/{output_file}'
                print('[INFO] Saving to final output location')
            print(f'[INFO] Merged forecast will be saved to: {output_file}')

            # Load and merge datasets sequentially along init_date dimension
            print('[INFO] Loading and merging forecast chunks...')
            merged_ds = None

            for i, file in enumerate(chunk_files):
                try:
                    print(f'[INFO] Processing file {i+1}/{len(chunk_files)}: '
                          f'{os.path.basename(file)}')
                    current_ds = xr.open_dataset(file, decode_timedelta=True)
                    if merged_ds is None:
                        merged_ds = current_ds
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
                    print(f'[ERROR] Could not process {file}: {e}')
                    return False

            if merged_ds is None:
                print('[ERROR] No valid forecast chunks could be loaded')
                return False
            print(f'[INFO] Successfully merged {len(chunk_files)} datasets')

            # Sort by init_date and add exclusion metadata
            merged_ds = merged_ds.sortby('init_date')
            print(f'[INFO] Sorted {len(merged_ds.init_date)} init dates in '
                  f'chronological order')
            if 'init_date' in merged_ds['grid_weights'].dims:
                merged_ds['grid_weights'] = merged_ds['grid_weights'].isel(
                    init_date=0, drop=True)
            self._add_exclusion_metadata(merged_ds, self.config)

            # Save with compression
            print(f'[INFO] Saving merged forecast to {output_file}...')
            encoding = {}
            for var in merged_ds.variables:
                encoding[var] = {'zlib': True, 'complevel': 2}
                dtype_nm = str(merged_ds[var].dtype)
                if 'float64' in dtype_nm:
                    encoding[var]['dtype'] = 'float32'

            merged_ds.to_netcdf(output_file, encoding=encoding)
            print(f'[SUCCESS] Merged forecast saved to {output_file}')

            # Close dataset
            merged_ds.close()

            # Delete chunk files after successful merge
            print('[INFO] Deleting chunk files after successful merge...')
            deleted_count = 0
            for file in chunk_files:
                try:
                    os.remove(file)
                    deleted_count += 1
                    print(f'  [OK] Deleted: {os.path.basename(file)}')
                except Exception as e:
                    print(f'  [ERROR] Could not delete {file}: {e}')

            print(f'[INFO] Deleted {deleted_count}/{len(chunk_files)} '
                  f'chunk files')
            return True

        except Exception as e:
            print(f'[ERROR] Failed to merge forecast chunks: {e}')
            traceback.print_exc()
            return False

    def merge_collection_datasets(self, dataset_type: str, info_dir: str
                                  ) -> bool:
        '''
        Merge collection-specific dataset files and calculate missing variables
        Parameters:
        dataset_type: type of dataset to merge ('fcst', 'ana', 'clim')
        info_dir: directory containing the tmp folder with collection files
        Returns: True if merge was successful, False otherwise
        '''
        print('\n==================================================')
        print(f'MERGING COLLECTION DATASETS: {dataset_type.upper()}')
        print('==================================================')

        # Find and organize collection files
        pattern = f'outputs/{info_dir}/tmp/{dataset_type}_*_*.nc4'
        print(f'[INFO] Looking for collection files matching: {pattern}')
        coll_files = sorted(glob.glob(pattern))
        if not coll_files:
            print(f'[ERROR] No collection files found for {dataset_type}')
            return False
        print(f'[INFO] Found {len(coll_files)} collection files:')
        for file in coll_files:
            print(f'  - {os.path.basename(file)}')

        try:
            # Generate output filename
            output_file = self._generate_output_filenm(dataset_type)
            output_file = f'outputs/{output_file}'
            print(f'[INFO] Merged dataset will be saved to: {output_file}')

            # Load and merge datasets sequentially
            print('[INFO] Loading and merging collection datasets...')
            merged_ds = None

            for i, file in enumerate(coll_files):
                try:
                    print(f'[INFO] Processing file {i+1}/{len(coll_files)}: '
                          f'{os.path.basename(file)}')
                    current_ds = xr.open_dataset(file, decode_timedelta=True)
                    if merged_ds is None:
                        # First dataset becomes the base
                        merged_ds = current_ds
                        print(f'[INFO] Loaded base dataset with variables: '
                              f'{list(merged_ds.data_vars.keys())}')
                    else:
                        # Merge with existing (xarray removes duplicates)
                        merged_ds = xr.merge([merged_ds, current_ds],
                                             compat='override')
                        print(f'[INFO] Merged dataset now has variables: '
                              f'{list(merged_ds.data_vars.keys())}')
                        # Close the current dataset to free memory
                        current_ds.close()
                        del current_ds
                    # Force garbage collection after each merge
                    gc.collect()
                except Exception as e:
                    print(f'[ERROR] Could not process {file}: {e}')
                    return False
            if merged_ds is None:
                print('[ERROR] No valid collection datasets could be loaded')
                return False
            print(f'[INFO] Successfully merged {len(coll_files)} datasets')

            # Calculate missing variables
            print('[INFO] Checking for calculable variables...')

            # Get originally requested calculable variables across collections
            all_requested_calc_vars = set()
            for coll_nm in self.original_var_colls:
                coll_vars = self.original_var_colls[coll_nm]
                needed_vars = coll_vars['3d'] + coll_vars['2d']
                for calc_var in CALCULABLE_VARS.keys():
                    if calc_var in needed_vars:
                        all_requested_calc_vars.add(calc_var)
            if all_requested_calc_vars:
                print(f'[INFO] Found calculable variables to process: '
                      f'{list(all_requested_calc_vars)}')
                # Check which calculable variables are missing
                missing_calc_vars = {}
                for calc_var in all_requested_calc_vars:
                    if calc_var not in merged_ds.data_vars:
                        dependencies = CALCULABLE_VARS[calc_var]
                        accumulated_deps = {}
                        # Check if all dependencies are available
                        for dep_var in dependencies:
                            if dep_var in merged_ds.data_vars:
                                accumulated_deps[dep_var] = {
                                    'collection': 'merged',
                                    'alias': dep_var
                                }
                        # Only add if have all dependencies
                        if len(accumulated_deps) == len(dependencies):
                            missing_calc_vars[calc_var] = accumulated_deps
                            print(f'[INFO] Can calculate {calc_var} from '
                                  f'dependencies: '
                                  f'{list(accumulated_deps.keys())}')
                        else:
                            missing_deps = [dep for dep in dependencies
                                            if dep not in accumulated_deps]
                            print(f'[ERROR] Cannot calculate {calc_var}: '
                                  f'missing dependencies {missing_deps}')
                            print(f'[ERROR] Available variables: '
                                  f'{list(merged_ds.data_vars.keys())}')
                            print(f'[ERROR] Searched in {len(coll_files)} '
                                  f'collection files')
                            return False
                # Calculate missing variables
                if missing_calc_vars:
                    print(f'[INFO] Calculating {len(missing_calc_vars)} '
                          f'variables...')
                    merged_ds = self._calculate_variables(merged_ds,
                                                          missing_calc_vars)

            # Filter to final variables and add scaling
            print('[INFO] Filtering to requested variables...')
            # Create mock validation results for filtering
            mock_val_results = {dataset_type: {'coll_results': {'merged': {
                'found_vars': {var: var for var in merged_ds.data_vars.keys()},
                'calculated_vars': {}}}}}

            # Filter dataset to requested variables, grid_weights, and coords
            dimension_lookup = {
                'ana':  ['time', 'lev', 'lat', 'lon'],
                'clim': ['time', 'lev', 'lat', 'lon'],
                'fcst': ['init_date', 'lead', 'lev', 'lat', 'lon']
            }
            filtered_ds = self._clean_and_filter_dataset(
                merged_ds, dimension_lookup[dataset_type], dataset_type,
                mock_val_results, skip_calc_mode=False
            )

            # Add scaling and metadata
            print('[INFO] Applying variable scaling and metadata...')
            for var_nms in filtered_ds.data_vars:
                if var_nms != 'grid_weights':
                    var_upper = var_nms.upper()
                    # Apply scaling
                    if var_upper in VARS_SCALE_MAP:
                        scale_factor = VARS_SCALE_MAP[var_upper]
                        if scale_factor != 1.0:
                            filtered_ds[var_nms] = filtered_ds[var_nms
                                                               ] * scale_factor
                            print(f'[INFO] Applied scaling factor '
                                  f'{scale_factor} to {var_nms}')
                    # Add metadata
                    if var_upper in VARS_LONG_MAP:
                        filtered_ds[var_nms].attrs['long_name'
                                                   ] = VARS_LONG_MAP[var_upper]
                    if var_upper in VARS_UNIT_MAP:
                        filtered_ds[var_nms].attrs['units'
                                                   ] = VARS_UNIT_MAP[var_upper]

            # Save merged dataset
            print(f'[INFO] Saving merged dataset to {output_file}...')
            # Save with compression
            encoding = {}
            for var in filtered_ds.variables:
                encoding[var] = {'zlib': True, 'complevel': 2}
                dtype_nm = str(filtered_ds[var].dtype)
                if 'float64' in dtype_nm:
                    encoding[var]['dtype'] = 'float32'
            # Add exclusion metadata if this is a forecast dataset
            if dataset_type == 'fcst':
                self._add_exclusion_metadata(filtered_ds, self.config)
            filtered_ds.to_netcdf(output_file, encoding=encoding)
            # Print summary
            print(f'[SUCCESS] Merged dataset saved to {output_file}')
            print(f'[INFO] Final variables: '
                  f'{list(filtered_ds.data_vars.keys())}')
            print(f'[INFO] Final dimensions: '
                  f'{dict(filtered_ds.sizes)}')
            # Close datasets
            merged_ds.close()
            filtered_ds.close()

            # Delete collection files after successful merge
            print('[INFO] Deleting collection files after successful merge...')
            deleted_count = 0
            for file in coll_files:
                try:
                    os.remove(file)
                    deleted_count += 1
                    print(f'  [OK] Deleted: {os.path.basename(file)}')
                except Exception as e:
                    print(f'  [ERROR] Could not delete {file}: {e}')

            print(f'[INFO] Deleted {deleted_count}/{len(coll_files)} '
                  f'collection files')
            return True
        except Exception as e:
            print(f'[ERROR] Failed to merge collections: {e}')
            traceback.print_exc()
            return False

