'''Shared metadata and constants for the stats processing workflow.'''

# ================== STATS INFO ==================

STATS_TO_CALCULATE_REGIONAL = ['acorr', 'rms']
STATS_TO_CALCULATE_GLOBAL = ['f', 'acorr', 'rms']

STATS_LONG_NAMES = {'acorr':   'Anomaly Correlation',
                    'rms':     'Root Mean Square Error',
                    'rms_ran': 'Random Component of RMS Error',
                    'rms_bar': 'Bias Component of RMS Error',
                    'rms_dis': 'Dissipation Component of RMS Error',
                    'rms_dsp': 'Dispersion Component of RMS Error',
                    'rms_amp': 'Amplitude Component of RMS Error',
                    'rms_phz': 'Phase Component of RMS Error',
                    'mse':     'Mean Square Error',
                    'me':      'Mean Error',
                    'res1':    'Residual from bar/amp/phz decomp',
                    'res2':    'Residual from bar/random decomp',
                    'f':       'fcst',
                    'a':       'ana',
                    'f_c':     'Forecast Minus Climatology'
                    }

# Define exceptions for units for stats
SQUARED_UNIT_STATS = ['mse', 'rms_dis', 'rms_dsp']
UNITLESS_STATS = ['acorr']

# ================== VARIABLE METADATA ==================

ALL_COLL_NMS = ['default', 'slices', 'aerosol']

VARS_LONG_MAP = {'H': 'Heights', 'U': 'U-Wind', 'V': 'V-Wind',
                 'T': 'Temperature', 'Q': 'Specific Humidity',
                 'P': 'Sea-Level Pressure', 'PS': 'Surface Pressure',
                 'Q2M': '2m Specific Humidity', 'T2M': '2m Temperature',
                 'U10M': '10m U-Wind', 'V10M': '10m V-Wind',
                 'D2M': '2m Dew Point',
                 'W': 'Vertical Velocity',  # ADD THIS
                 'Z': 'Geopotential',
                 'AOD': 'Total Aerosol Extinction AOT [550 nm]',
                 'LOGAOD': 'log(AOD+0.01)', 'PM25': 'PM2.5 Total Mass'}

VARS_UNIT_MAP = {'H': 'm', 'U': 'm/s', 'V': 'm/s', 'T': 'K', 'Q': 'g/kg',
                 'P': 'hPa', 'PS': 'hPa', 'Q2M': 'g/kg', 'T2M': 'K',
                 'U10M': 'm/s', 'V10M': 'm/s', 'D2M': 'K',
                 'W': 'Pa/s',  # ADD THIS (or 'm/s' depending on your data)
                 'Z': 'mÂ²/sâ»Â²',
                 'AOD': '',
                 'LOGAOD': '', 'PM25': 'Âµg/m3'}

VARS_SCALE_MAP = {'H': 1., 'U': 1., 'V': 1., 'T': 1., 'Q': 1000., 'P': 1./100.,
                  'PS': 1./100., 'Q2M': 1000., 'T2M': 1., 'U10M': 1.,
                  'V10M': 1., 'D2M': 1.,
                  'W': 1.,  # ADD THIS (adjust if your data needs scaling)
                  'Z': 1.,
                  'AOD': 1., 'LOGAOD': 1., 'PM25': 1E6}

# Calculable variables and their dependencies
# Note: see _calculate_variables() method for equations
CALCULABLE_VARS = {'D2m': ['Q2m', 'PS'], 'LOGAOD': ['AOD'],
                   'Z': ['H'],
                   'PM25': ['SSSMASS25', 'DUSMASS25', 'BCSMASS', 'OCSMASS',
                            'SO4SMASS', 'NISMASS25', 'NH4SMASS']}

PRITHVI_PRESSURE_LVLS = {
    72: 985,
    71: 970,
    68: 925,
    63: 850,
    56: 700,
    53: 600,
    51: 525,
    48: 412,
    45: 288,
    44: 245,
    43: 208,
    41: 150,
    39: 109,
    34: 48
}

# ================== REGIONS METADATA (FOR SCORECARDS) ==================
# Note: see _create_regional_masks() method for region definitions for stats

REGION_SHORT_MAP = {'GLO': 'global',  'NHE': 'n.hem',   'TRO': 'tropics',
                    'SHE': 's.hem',   'NWQ': 'nw.quad', 'NEQ': 'ne.quad',
                    'SWQ': 'sw.quad', 'SEQ': 'se.quad', 'NAM': 'america',
                    'EUR': 'europe',  'NPO': 'npolar',  'SPO': 'spolar',
                    'XPO': 'xpolar',  'CUS': 'conus',   'LND': 'glob_l',
                    'NHL': 'n.hem_l', 'TRL': 'trop_l',  'SHL': 's.hem_l',}

REGION_COORDS_MAP = {
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
    'LND': [90.0, -90.0, 180.0, -180.0],
    'NHL': [80.0, 20.0, 180.0, -180.0],
    'TRL': [20.0, -20.0, 180.0, -180.0],
    'SHL': [-20.0, -80.0, 180.0, -180.0],
}

# ================== PRITHVI LEVEL MAPPING HELPER ======================


def map_pressure_to_prithvi_levels(
    requested_pressures, prithvi_dict=PRITHVI_PRESSURE_LVLS
):
    """
    Map requested pressure levels to closest available Prithvi model levels.

    Returns:
    --------
    tuple of (prithvi_levels, pressure_mapping)
        prithvi_levels : list of Prithvi model level indices
        pressure_mapping : dict {prithvi_level: requested_pressure}
    """
    if not requested_pressures:
        return None, {}

    prithvi_levels = []
    pressure_mapping = {}

    for requested_p in requested_pressures:
        # Find closest pressure value in Prithvi dict
        closest_prithvi_lev = min(
            prithvi_dict.keys(),
            key=lambda k: abs(prithvi_dict[k] - requested_p)
        )
        actual_pressure = prithvi_dict[closest_prithvi_lev]

        # Warn if difference is large
        diff = abs(actual_pressure - requested_p)
        if diff > 50:  # More than 50 hPa difference
            print(f'  [WARNING] Large pressure mismatch: requested {requested_p} hPa, '
                  f'using Prithvi level {closest_prithvi_lev} ({actual_pressure} hPa), '
                  f'difference = {diff} hPa')

        if closest_prithvi_lev not in prithvi_levels:
            prithvi_levels.append(closest_prithvi_lev)
            pressure_mapping[closest_prithvi_lev] = requested_p

    return prithvi_levels, pressure_mapping

