'''YAML configuration helpers.'''

from collections import OrderedDict
from dataclasses import dataclass

import yaml

# ================== YAML CONFIGURATION CLASSES ==================


class DuplicateKeysError(Exception):
    pass


class SafeLoaderWithDuplicateKeyCheck(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise DuplicateKeysError(
                    f'[ERROR] DUPLICATE KEY DETECTED: "{key}"\n'
                    f'This key appears multiple times in your YAML file.\n'
                    f'Please remove the duplicate entry.')
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping


@dataclass
class RuntimeSettings:
    '''Resolved runtime settings with CLI > YAML > defaults precedence.'''
    stats_types: str
    pipeline_fail_policy: str
    pipeline_branch_execution: str
    pipeline_resume_mode: str
    pipeline_summary_file: str
    pipeline_max_workers_dataset: int
    pipeline_max_workers_dataset_fcst: int
    pipeline_max_workers_dataset_ana: int
    pipeline_max_workers_dataset_clim: int
    pipeline_chunk_size_fcst: int
    pipeline_chunk_size_ana: int
    pipeline_chunk_size_clim: int
    pipeline_max_workers_stats: int
    pipeline_max_workers_stats_regional: int
    pipeline_max_workers_stats_global: int
    pipeline_chunk_size_stats: int


def _resolve_setting(cli_value, yaml_value, default_value):
    '''Resolve one setting value using CLI > YAML > default precedence.'''
    if cli_value is not None:
        return cli_value
    if yaml_value is not None:
        return yaml_value
    return default_value


def resolve_runtime_settings(args, config):
    '''Resolve pipeline runtime settings from CLI args and loaded config.'''
    stats_types = _resolve_setting(
        getattr(args, 'stats_types', None),
        config.get('stats_types'),
        'both',
    )
    pipeline_fail_policy = _resolve_setting(
        getattr(args, 'pipeline_fail_policy', None),
        config.get('pipeline_fail_policy'),
        'partial_ok',
    )
    pipeline_branch_execution = _resolve_setting(
        getattr(args, 'pipeline_branch_execution', None),
        config.get('pipeline_branch_execution'),
        'parallel',
    )
    pipeline_resume_mode = _resolve_setting(
        getattr(args, 'pipeline_resume_mode', None),
        config.get('pipeline_resume_mode'),
        'safe',
    )
    pipeline_summary_file = _resolve_setting(
        getattr(args, 'pipeline_summary_file', None),
        config.get('pipeline_summary_file'),
        'run_summary.txt',
    )
    pipeline_max_workers_dataset = _resolve_setting(
        getattr(args, 'pipeline_max_workers_dataset', None),
        config.get('pipeline_max_workers_dataset'),
        None,
    )
    pipeline_max_workers_dataset_fcst = _resolve_setting(
        getattr(args, 'pipeline_max_workers_dataset_fcst', None),
        config.get('pipeline_max_workers_dataset_fcst'),
        pipeline_max_workers_dataset if pipeline_max_workers_dataset else 8,
    )
    pipeline_max_workers_dataset_ana = _resolve_setting(
        getattr(args, 'pipeline_max_workers_dataset_ana', None),
        config.get('pipeline_max_workers_dataset_ana'),
        pipeline_max_workers_dataset if pipeline_max_workers_dataset else 8,
    )
    pipeline_max_workers_dataset_clim = _resolve_setting(
        getattr(args, 'pipeline_max_workers_dataset_clim', None),
        config.get('pipeline_max_workers_dataset_clim'),
        pipeline_max_workers_dataset if pipeline_max_workers_dataset else 4,
    )
    pipeline_chunk_size_fcst = _resolve_setting(
        getattr(args, 'pipeline_chunk_size_fcst', None),
        config.get('pipeline_chunk_size_fcst'),
        1,
    )
    pipeline_chunk_size_ana = _resolve_setting(
        getattr(args, 'pipeline_chunk_size_ana', None),
        config.get('pipeline_chunk_size_ana'),
        4,
    )
    pipeline_chunk_size_clim = _resolve_setting(
        getattr(args, 'pipeline_chunk_size_clim', None),
        config.get('pipeline_chunk_size_clim'),
        4,
    )
    pipeline_max_workers_stats = _resolve_setting(
        getattr(args, 'pipeline_max_workers_stats', None),
        config.get('pipeline_max_workers_stats'),
        None,
    )
    pipeline_max_workers_stats_regional = _resolve_setting(
        getattr(args, 'pipeline_max_workers_stats_regional', None),
        config.get('pipeline_max_workers_stats_regional'),
        pipeline_max_workers_stats if pipeline_max_workers_stats else 4,
    )
    pipeline_max_workers_stats_global = _resolve_setting(
        getattr(args, 'pipeline_max_workers_stats_global', None),
        config.get('pipeline_max_workers_stats_global'),
        pipeline_max_workers_stats if pipeline_max_workers_stats else 4,
    )
    pipeline_chunk_size_stats = _resolve_setting(
        getattr(args, 'pipeline_chunk_size_stats', None),
        config.get('pipeline_chunk_size_stats'),
        2,
    )

    valid_stats_types = {'regional', 'global', 'both'}
    valid_fail_policy = {'fail_fast', 'partial_ok'}
    valid_branch_execution = {'parallel', 'sequential'}
    valid_resume_mode = {'off', 'safe'}

    if stats_types not in valid_stats_types:
        raise ValueError(
            f'Invalid stats_types: {stats_types}. '
            f'Expected one of: {sorted(valid_stats_types)}'
        )
    if pipeline_fail_policy not in valid_fail_policy:
        raise ValueError(
            f'Invalid pipeline_fail_policy: {pipeline_fail_policy}. '
            f'Expected one of: {sorted(valid_fail_policy)}'
        )
    if pipeline_branch_execution not in valid_branch_execution:
        raise ValueError(
            f'Invalid pipeline_branch_execution: {pipeline_branch_execution}. '
            f'Expected one of: {sorted(valid_branch_execution)}'
        )
    if pipeline_resume_mode not in valid_resume_mode:
        raise ValueError(
            f'Invalid pipeline_resume_mode: {pipeline_resume_mode}. '
            f'Expected one of: {sorted(valid_resume_mode)}'
        )
    if not pipeline_summary_file or not str(pipeline_summary_file).strip():
        raise ValueError('pipeline_summary_file cannot be empty')

    if pipeline_max_workers_dataset is not None:
        try:
            pipeline_max_workers_dataset = int(pipeline_max_workers_dataset)
        except (TypeError, ValueError):
            raise ValueError('pipeline_max_workers_dataset must be an integer')
        if pipeline_max_workers_dataset < 1:
            raise ValueError('pipeline_max_workers_dataset must be >= 1')

    try:
        pipeline_max_workers_dataset_fcst = int(
            pipeline_max_workers_dataset_fcst)
    except (TypeError, ValueError):
        raise ValueError(
            'pipeline_max_workers_dataset_fcst must be an integer')
    if pipeline_max_workers_dataset_fcst < 1:
        raise ValueError('pipeline_max_workers_dataset_fcst must be >= 1')

    try:
        pipeline_max_workers_dataset_ana = int(
            pipeline_max_workers_dataset_ana)
    except (TypeError, ValueError):
        raise ValueError(
            'pipeline_max_workers_dataset_ana must be an integer')
    if pipeline_max_workers_dataset_ana < 1:
        raise ValueError('pipeline_max_workers_dataset_ana must be >= 1')

    try:
        pipeline_max_workers_dataset_clim = int(
            pipeline_max_workers_dataset_clim)
    except (TypeError, ValueError):
        raise ValueError(
            'pipeline_max_workers_dataset_clim must be an integer')
    if pipeline_max_workers_dataset_clim < 1:
        raise ValueError('pipeline_max_workers_dataset_clim must be >= 1')

    try:
        pipeline_chunk_size_fcst = int(pipeline_chunk_size_fcst)
    except (TypeError, ValueError):
        raise ValueError('pipeline_chunk_size_fcst must be an integer')
    if pipeline_chunk_size_fcst < 1:
        raise ValueError('pipeline_chunk_size_fcst must be >= 1')

    try:
        pipeline_chunk_size_ana = int(pipeline_chunk_size_ana)
    except (TypeError, ValueError):
        raise ValueError('pipeline_chunk_size_ana must be an integer')
    if pipeline_chunk_size_ana < 1:
        raise ValueError('pipeline_chunk_size_ana must be >= 1')

    try:
        pipeline_chunk_size_clim = int(pipeline_chunk_size_clim)
    except (TypeError, ValueError):
        raise ValueError('pipeline_chunk_size_clim must be an integer')
    if pipeline_chunk_size_clim < 1:
        raise ValueError('pipeline_chunk_size_clim must be >= 1')

    if pipeline_max_workers_stats is not None:
        try:
            pipeline_max_workers_stats = int(pipeline_max_workers_stats)
        except (TypeError, ValueError):
            raise ValueError('pipeline_max_workers_stats must be an integer')
        if pipeline_max_workers_stats < 1:
            raise ValueError('pipeline_max_workers_stats must be >= 1')

    try:
        pipeline_max_workers_stats_regional = int(
            pipeline_max_workers_stats_regional)
    except (TypeError, ValueError):
        raise ValueError(
            'pipeline_max_workers_stats_regional must be an integer')
    if pipeline_max_workers_stats_regional < 1:
        raise ValueError('pipeline_max_workers_stats_regional must be >= 1')

    try:
        pipeline_max_workers_stats_global = int(
            pipeline_max_workers_stats_global)
    except (TypeError, ValueError):
        raise ValueError(
            'pipeline_max_workers_stats_global must be an integer')
    if pipeline_max_workers_stats_global < 1:
        raise ValueError('pipeline_max_workers_stats_global must be >= 1')

    try:
        pipeline_chunk_size_stats = int(pipeline_chunk_size_stats)
    except (TypeError, ValueError):
        raise ValueError('pipeline_chunk_size_stats must be an integer')
    if pipeline_chunk_size_stats < 1:
        raise ValueError('pipeline_chunk_size_stats must be >= 1')

    return RuntimeSettings(
        stats_types=stats_types,
        pipeline_fail_policy=pipeline_fail_policy,
        pipeline_branch_execution=pipeline_branch_execution,
        pipeline_resume_mode=pipeline_resume_mode,
        pipeline_summary_file=pipeline_summary_file,
        pipeline_max_workers_dataset=pipeline_max_workers_dataset,
        pipeline_max_workers_dataset_fcst=pipeline_max_workers_dataset_fcst,
        pipeline_max_workers_dataset_ana=pipeline_max_workers_dataset_ana,
        pipeline_max_workers_dataset_clim=pipeline_max_workers_dataset_clim,
        pipeline_chunk_size_fcst=pipeline_chunk_size_fcst,
        pipeline_chunk_size_ana=pipeline_chunk_size_ana,
        pipeline_chunk_size_clim=pipeline_chunk_size_clim,
        pipeline_max_workers_stats=pipeline_max_workers_stats,
        pipeline_max_workers_stats_regional=(
            pipeline_max_workers_stats_regional),
        pipeline_max_workers_stats_global=pipeline_max_workers_stats_global,
        pipeline_chunk_size_stats=pipeline_chunk_size_stats,
    )
