'''Dataset regridding helper.'''

import warnings

warnings.filterwarnings(
    'ignore',
    message='Latitude is outside of \\[-90, 90\\]',
    category=UserWarning,
    module='xesmf.backend',
)

import numpy as np
import xarray as xr
import xesmf as xe


class DatasetRegridder:
    '''Own xESMF regridder caching and input preparation.'''

    def __init__(self, target_grid):
        self.target_grid = target_grid
        self.regridders = {}

    def get_regridder(self, source_ds, grid_type):
        '''Retrieve or create regridder for given source dataset.'''
        lat_key = source_ds.lat.shape[0] if 'lat' in source_ds.dims else 0
        lon_key = source_ds.lon.shape[0] if 'lon' in source_ds.dims else 0
        regridder_key = f'{grid_type}_{lat_key}x{lon_key}'
        if regridder_key not in self.regridders:
            print(f'    Creating regridder for {grid_type}')
            try:
                # Use conservative_normed for correct polar handling.
                regridder = xe.Regridder(
                    source_ds, self.target_grid, 'conservative_normed',
                    periodic=True)
                self.regridders[regridder_key] = regridder
            except Exception as e:
                print(f'    [ERROR] Regridder creation failed: {e}')
                return None
        return self.regridders[regridder_key]

    def make_input_contiguous(self, data_array):
        '''Return a C-contiguous DataArray for xESMF regridding.'''
        values = data_array.data
        try:
            if hasattr(values, 'flags') and values.flags['C_CONTIGUOUS']:
                return data_array
            contiguous_values = np.ascontiguousarray(values)
            return xr.DataArray(contiguous_values,
                                coords=data_array.coords,
                                dims=data_array.dims,
                                attrs=data_array.attrs,
                                name=data_array.name)
        except Exception:
            return data_array

    def _standardize_time_coord(self, regridded, std_coords):
        '''Apply caller-standardized time coordinates when present.'''
        if std_coords is not None and 'time' in regridded.coords:
            return regridded.assign_coords(time=std_coords)
        return regridded

    def regrid_variable_map(self, regridder, source_vars, std_coords=None):
        '''
        Regrid several variables in one xESMF Dataset call.

        Parameters:
        - regridder: xESMF Regridder instance.
        - source_vars: dict mapping output variable name to source DataArray.
        - std_coords: optional replacement time coordinate.

        Returns:
        - dict mapping output variable name to regridded DataArray.
        '''
        if not source_vars:
            return {}

        batch_vars = {}
        for target_var, data_array in source_vars.items():
            batch_vars[target_var] = self.make_input_contiguous(data_array)

        batch_ds = xr.Dataset(batch_vars)
        regridded_ds = regridder(batch_ds)
        regridded_ds = self._standardize_time_coord(regridded_ds, std_coords)
        return {
            target_var: regridded_ds[target_var]
            for target_var in source_vars.keys()
        }

    def regrid_single_variable(self, regridder, data_array, std_coords=None):
        '''Regrid one variable, preserving the previous per-variable behavior.'''
        regrid_input = self.make_input_contiguous(data_array)
        regridded_data = regridder(regrid_input)
        return self._standardize_time_coord(regridded_data, std_coords)
