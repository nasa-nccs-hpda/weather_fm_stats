import importlib
import importlib.machinery
import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def ensure_fake_dependencies():
    """Install lightweight fakes for optional HPC dependencies."""
    fake_packages = {}

    try:
        import metpy  # noqa: F401
    except ModuleNotFoundError:
        fake_metpy = types.ModuleType('metpy')
        fake_calc = types.ModuleType('metpy.calc')
        fake_units = types.ModuleType('metpy.units')
        fake_units.units = object()
        fake_metpy.calc = fake_calc
        fake_metpy.units = fake_units
        fake_metpy.__spec__ = importlib.machinery.ModuleSpec(
            'metpy', loader=None)
        fake_calc.__spec__ = importlib.machinery.ModuleSpec(
            'metpy.calc', loader=None)
        fake_units.__spec__ = importlib.machinery.ModuleSpec(
            'metpy.units', loader=None)
        fake_packages['metpy'] = fake_metpy
        fake_packages['metpy.calc'] = fake_calc
        fake_packages['metpy.units'] = fake_units

    try:
        import xesmf  # noqa: F401
    except ModuleNotFoundError:
        fake_xesmf = types.ModuleType('xesmf')
        fake_xesmf.Regridder = (
            lambda source, target, method, periodic=True: lambda data: data)
        fake_xesmf.__spec__ = importlib.machinery.ModuleSpec(
            'xesmf', loader=None)
        fake_packages['xesmf'] = fake_xesmf

    for package_name, module_obj in fake_packages.items():
        sys.modules[package_name] = module_obj


def import_batch_processor_class():
    ensure_fake_dependencies()
    module = importlib.import_module('model.dataset_processor')
    return module.BatchDatasetProcessor


def make_minimal_config():
    return {
        'fcst_model': 'TEST_FCST',
        'ana_model': 'TEST_ANA',
        'clim_model': 'TEST_CLIM',
        'expver': 1,
        'verify': 'Y',
        'levels': [1000, 850],
        'regions': ['GLO'],
        'Nlat': 3,
        'Nlon': 4,
        'delta_lat': 90.0,
        'delta_lon': 90.0,
        'fcst_input_dir': '',
        'ana_input_dir': '',
        'clim_input_dir': '',
        'dir_loc': [],
    }


def make_synthetic_dataset(num_3d_vars=4, num_2d_vars=3, time_name='time'):
    lat = np.array([-90.0, 0.0, 90.0])
    lon = np.array([-180.0, -90.0, 0.0, 90.0])
    time = pd.date_range('2000-01-01', periods=1)
    lev = np.array([1000, 850], dtype=np.int32)
    coords = {time_name: time, 'lev': lev, 'lat': lat, 'lon': lon}
    ds = xr.Dataset(coords=coords)

    for i in range(num_3d_vars):
        data = np.full((1, 2, 3, 4), fill_value=float(i), dtype=np.float32)
        ds[f'IN3D{i}'] = ((time_name, 'lev', 'lat', 'lon'), data)

    for i in range(num_2d_vars):
        data = np.full((1, 3, 4), fill_value=100.0 + i, dtype=np.float32)
        ds[f'IN2D{i}'] = ((time_name, 'lat', 'lon'), data)

    return ds


def make_validation_result(num_3d_vars=4, num_2d_vars=3):
    found_vars = {
        **{f'OUT3D{i}': f'IN3D{i}' for i in range(num_3d_vars)},
        **{f'OUT2D{i}': f'IN2D{i}' for i in range(num_2d_vars)},
    }
    return {
        'coll_results': {
            'default': {
                'found_vars': found_vars,
                'calculated_vars': {},
            }
        }
    }


def make_calculated_validation_result():
    return {
        'coll_results': {
            'slices': {
                'found_vars': {'D2m': 'calculated'},
                'calculated_vars': {
                    'D2m': {
                        'Q2m': {'collection': 'slices', 'alias': 'Q2m'},
                        'PS': {'collection': 'slices', 'alias': 'PS'},
                    }
                },
            }
        }
    }


def make_calculated_dependency_dataset():
    lat = np.array([-90.0, 0.0, 90.0])
    lon = np.array([-180.0, -90.0, 0.0, 90.0])
    time = pd.date_range('2000-01-01', periods=1)
    ds = xr.Dataset(coords={'time': time, 'lat': lat, 'lon': lon})
    ds['Q2m'] = (
        ('time', 'lat', 'lon'),
        np.full((1, 3, 4), 0.004, dtype=np.float32),
    )
    ds['PS'] = (
        ('time', 'lat', 'lon'),
        np.full((1, 3, 4), 100000.0, dtype=np.float32),
    )
    return ds


class IdentityRegridder:
    def __init__(self):
        self.calls = []

    def __call__(self, data):
        self.calls.append(type(data).__name__)
        return data


def force_processor_regridder(processor, regridder):
    processor.dataset_regridder.get_regridder = (
        lambda source_ds, grid_type: regridder)


def force_batch_failure(processor):
    def fail_batch(regridder, source_vars, std_coords=None):
        raise RuntimeError('synthetic batch failure')

    processor.dataset_regridder.regrid_variable_map = fail_batch


def process_with_processor(processor, ds, validation_result, colls=('default',)):
    return processor._process_dataset_variables(
        ds, list(colls), 'ana', validation_result)[0]


def assert_dataarray_maps_equal(left, right):
    assert set(left.keys()) == set(right.keys())
    for key in left:
        xr.testing.assert_equal(left[key], right[key])


def test_batch_and_fallback_paths_produce_equivalent_outputs():
    processor_class = import_batch_processor_class()
    ds = make_synthetic_dataset(num_3d_vars=5, num_2d_vars=4)
    validation_result = make_validation_result(num_3d_vars=5, num_2d_vars=4)

    batch_processor = processor_class(make_minimal_config())
    batch_regridder = IdentityRegridder()
    force_processor_regridder(batch_processor, batch_regridder)

    fallback_processor = processor_class(make_minimal_config())
    fallback_regridder = IdentityRegridder()
    force_processor_regridder(fallback_processor, fallback_regridder)
    force_batch_failure(fallback_processor)

    batch_vars = process_with_processor(batch_processor, ds, validation_result)
    fallback_vars = process_with_processor(
        fallback_processor, ds, validation_result)

    assert_dataarray_maps_equal(batch_vars, fallback_vars)
    assert batch_regridder.calls == ['Dataset']
    assert fallback_regridder.calls == ['DataArray'] * 9


def test_batch_path_uses_one_regridder_call_for_mixed_2d_and_3d_variables():
    processor_class = import_batch_processor_class()
    processor = processor_class(make_minimal_config())
    regridder = IdentityRegridder()
    force_processor_regridder(processor, regridder)

    ds = make_synthetic_dataset(num_3d_vars=6, num_2d_vars=6)
    validation_result = make_validation_result(num_3d_vars=6, num_2d_vars=6)

    output_vars = process_with_processor(processor, ds, validation_result)

    assert len(output_vars) == 12
    assert regridder.calls == ['Dataset']


def test_batch_path_normalizes_nonstandard_time_coordinate():
    processor_class = import_batch_processor_class()
    processor = processor_class(make_minimal_config())
    regridder = IdentityRegridder()
    force_processor_regridder(processor, regridder)

    ds = make_synthetic_dataset(
        num_3d_vars=1, num_2d_vars=1, time_name='valid_time')
    validation_result = make_validation_result(num_3d_vars=1, num_2d_vars=1)

    output_vars = process_with_processor(processor, ds, validation_result)

    for data_array in output_vars.values():
        assert 'time' in data_array.coords
        np.testing.assert_array_equal(data_array['time'].values, np.array([0]))


def test_calculated_variable_dependencies_are_regridded_but_not_calculated():
    processor_class = import_batch_processor_class()
    processor = processor_class(make_minimal_config())
    regridder = IdentityRegridder()
    force_processor_regridder(processor, regridder)

    output_vars = process_with_processor(
        processor,
        make_calculated_dependency_dataset(),
        make_calculated_validation_result(),
        colls=('slices',),
    )

    assert set(output_vars.keys()) == {'Q2m', 'PS'}
    assert 'D2m' not in output_vars
    assert regridder.calls == ['Dataset']


def test_calculated_dependency_failure_still_raises_after_batch_fallback():
    processor_class = import_batch_processor_class()
    processor = processor_class(make_minimal_config())
    force_batch_failure(processor)

    class FailingSingleRegridder(IdentityRegridder):
        def __call__(self, data):
            self.calls.append(type(data).__name__)
            if getattr(data, 'name', None) == 'Q2m':
                raise RuntimeError('cannot regrid Q2m')
            return data

    force_processor_regridder(processor, FailingSingleRegridder())

    with pytest.raises(ValueError, match='Failed to process dependency Q2m'):
        process_with_processor(
            processor,
            make_calculated_dependency_dataset(),
            make_calculated_validation_result(),
            colls=('slices',),
        )


def test_processor_can_import_and_initialize():
    processor_class = import_batch_processor_class()
    proc = processor_class(make_minimal_config())
    assert proc is not None
