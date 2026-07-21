import importlib
import importlib.util
import importlib.machinery
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def ensure_fake_dependencies():
    """Install lightweight fake modules for optional dependencies missing from the runtime."""
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
        fake_metpy.__spec__ = importlib.machinery.ModuleSpec('metpy', loader=None)
        fake_calc.__spec__ = importlib.machinery.ModuleSpec('metpy.calc', loader=None)
        fake_units.__spec__ = importlib.machinery.ModuleSpec('metpy.units', loader=None)
        fake_packages['metpy'] = fake_metpy
        fake_packages['metpy.calc'] = fake_calc
        fake_packages['metpy.units'] = fake_units

    try:
        import xesmf  # noqa: F401
    except ModuleNotFoundError:
        fake_xesmf = types.ModuleType('xesmf')
        fake_xesmf.Regridder = lambda source, target, method, periodic=True: (lambda data: data)
        fake_xesmf.__spec__ = importlib.machinery.ModuleSpec('xesmf', loader=None)
        fake_packages['xesmf'] = fake_xesmf

    for package_name, module_obj in fake_packages.items():
        sys.modules[package_name] = module_obj


def import_new_batch_processor_class():
    ensure_fake_dependencies()
    module = importlib.import_module('model.dataset_processor')
    return module.BatchDatasetProcessor


def load_archived_batch_processor_class():
    repo_root = Path(__file__).resolve().parents[1]
    archive_path = repo_root / 'archives' / 'stats.py'
    spec = importlib.util.spec_from_file_location('archives_stats', str(archive_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
        'dir_loc': []
    }


def make_synthetic_dataset(num_vars=20, include_3d=True):
    lat = np.array([-90.0, 0.0, 90.0])
    lon = np.array([-180.0, -90.0, 0.0, 90.0])
    time = pd.date_range('2000-01-01', periods=1)
    lev = np.array([1000, 850], dtype=np.int32)
    coords = {'time': time, 'lev': lev, 'lat': lat, 'lon': lon}
    ds = xr.Dataset(coords=coords)

    for i in range(num_vars):
        var_name = f'IN{i}'
        data = np.full((1, 2, 3, 4), fill_value=float(i), dtype=np.float32)
        ds[var_name] = (('time', 'lev', 'lat', 'lon'), data)
    return ds


def make_validation_result(num_vars=20):
    found_vars = {f'OUT{i}': f'IN{i}' for i in range(num_vars)}
    return {
        'coll_results': {
            'default': {
                'found_vars': found_vars,
                'calculated_vars': {}
            }
        }
    }


def benchmark_process(processor, ds, validation_result, iterations=20):
    start = time.perf_counter()
    for _ in range(iterations):
        processor._process_dataset_variables(ds, ['default'], 'ana', validation_result)
    return (time.perf_counter() - start) / iterations


def test_new_and_old_batch_processing_produce_equivalent_outputs():
    """The new processor should produce the same variable outputs as the archived processor."""
    new_class = import_new_batch_processor_class()
    old_class = load_archived_batch_processor_class()
    config = make_minimal_config()

    new_proc = new_class(config.copy())
    old_proc = old_class(config.copy())

    # Override the regridder to avoid xESMF dependency and compare core logic
    dummy_regridder = lambda data: data
    new_proc._get_regridder = lambda source_ds, grid_type, collections=None: dummy_regridder
    old_proc._get_regridder = lambda source_ds, grid_type, collections=None: dummy_regridder

    ds = make_synthetic_dataset(num_vars=20)
    validation_result = make_validation_result(num_vars=20)

    new_vars, _ = new_proc._process_dataset_variables(ds, ['default'], 'ana', validation_result)
    old_vars, _ = old_proc._process_dataset_variables(ds, ['default'], 'ana', validation_result)

    assert set(new_vars.keys()) == set(old_vars.keys())
    for key in new_vars.keys():
        xr.testing.assert_equal(new_vars[key], old_vars[key])


def test_new_batch_path_uses_single_regridder_call_for_many_variables():
    """The new processor should invoke the regridder once while the old per-variable path invokes it repeatedly."""
    class CallCountingRegridder:
        def __init__(self):
            self.count = 0

        def __call__(self, data):
            self.count += 1
            return data

    new_class = import_new_batch_processor_class()
    old_class = load_archived_batch_processor_class()
    config = make_minimal_config()

    new_proc = new_class(config.copy())
    old_proc = old_class(config.copy())

    new_regridder = CallCountingRegridder()
    old_regridder = CallCountingRegridder()
    new_proc._get_regridder = lambda source_ds, grid_type, collections=None: new_regridder
    old_proc._get_regridder = lambda source_ds, grid_type, collections=None: old_regridder

    ds = make_synthetic_dataset(num_vars=40)
    validation_result = make_validation_result(num_vars=40)

    new_proc._process_dataset_variables(ds, ['default'], 'ana', validation_result)
    old_proc._process_dataset_variables(ds, ['default'], 'ana', validation_result)

    assert new_regridder.count == 1, (
        f'Batched processing should call the regridder once, but called it {new_regridder.count} times.')
    assert old_regridder.count == 40, (
        f'Per-variable processing should call the regridder once per variable, but called it {old_regridder.count} times.')


def test_archived_processor_can_import_and_initialize():
    """Sanity check that the archived processor is importable and can be initialized with a minimal config."""
    old_class = load_archived_batch_processor_class()
    config = make_minimal_config()
    proc = old_class(config)
    assert proc is not None


def test_new_processor_can_import_and_initialize():
    """Sanity check that the current processor is importable and can be initialized with a minimal config."""
    new_class = import_new_batch_processor_class()
    proc = new_class(make_minimal_config())
    assert proc is not None
