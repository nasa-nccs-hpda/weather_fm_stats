'''Deterministic chunk planning helpers.'''

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable, List


@dataclass
class ChunkSpec:
    '''One deterministic chunk of the configured init-date sequence.'''
    chunk_index: int
    start_idx: int
    end_idx: int
    all_dates: List[str]
    selected_dates: List[str]
    status: str
    output_path: str

    @property
    def chunk_id(self):
        return f'chunk_{self.start_idx:03d}_{self.end_idx:03d}'

    @property
    def is_skipped(self):
        return self.status == 'skipped'

    @property
    def is_required(self):
        return self.status != 'skipped'

    def to_dict(self):
        return asdict(self)


def _date_key(value):
    '''Return YYYYMMDD string for datetime/int/string date values.'''
    if isinstance(value, datetime):
        return value.strftime('%Y%m%d')
    value_str = str(value).strip()
    if '-' in value_str:
        return datetime.strptime(value_str, '%Y-%m-%d').strftime('%Y%m%d')
    return datetime.strptime(value_str, '%Y%m%d').strftime('%Y%m%d')


def _date_label(value):
    '''Return stable YYYY-MM-DD label for JSON/debug output.'''
    if isinstance(value, datetime):
        return value.strftime('%Y-%m-%d')
    return datetime.strptime(_date_key(value), '%Y%m%d').strftime('%Y-%m-%d')


class InitDateChunkPlanner:
    '''Build deterministic chunks from a full init-date sequence.'''

    def __init__(self, init_dates: Iterable[datetime], exclude_dates=None):
        self.init_dates = list(init_dates)
        self.exclude_keys = {
            _date_key(date_value) for date_value in (exclude_dates or [])
        }

    def build(self, chunk_size, output_dir, output_prefix):
        '''Return ChunkSpec objects in stable chunk_index order.'''
        chunks = []
        for chunk_index, start_idx in enumerate(
                range(0, len(self.init_dates), chunk_size)):
            end_idx = min(len(self.init_dates) - 1, start_idx + chunk_size - 1)
            all_dates = self.init_dates[start_idx:end_idx + 1]
            selected_dates = [
                date_value for date_value in all_dates
                if _date_key(date_value) not in self.exclude_keys
            ]
            chunk_id = f'chunk_{start_idx:03d}_{end_idx:03d}'
            output_name = f'{output_prefix}_{chunk_id}.nc4'
            chunks.append(ChunkSpec(
                chunk_index=chunk_index,
                start_idx=start_idx,
                end_idx=end_idx,
                all_dates=[_date_label(date_value) for date_value in all_dates],
                selected_dates=[
                    _date_label(date_value) for date_value in selected_dates
                ],
                status='required' if selected_dates else 'skipped',
                output_path=os.path.join(output_dir, output_name),
            ))
        return chunks


class SequenceChunkPlanner:
    '''Build deterministic chunks from an already ordered sequence.'''

    def __init__(self, items: Iterable, label_func=None):
        self.items = list(items)
        self.label_func = label_func or str

    def build(self, chunk_size, output_dir, output_prefix):
        '''Return ChunkSpec objects in stable chunk_index order.'''
        chunks = []
        for chunk_index, start_idx in enumerate(
                range(0, len(self.items), chunk_size)):
            end_idx = min(len(self.items) - 1, start_idx + chunk_size - 1)
            chunk_items = self.items[start_idx:end_idx + 1]
            labels = [self.label_func(item) for item in chunk_items]
            chunk_id = f'chunk_{start_idx:03d}_{end_idx:03d}'
            output_name = f'{output_prefix}_{chunk_id}.nc4'
            chunks.append(ChunkSpec(
                chunk_index=chunk_index,
                start_idx=start_idx,
                end_idx=end_idx,
                all_dates=labels,
                selected_dates=labels,
                status='required' if labels else 'skipped',
                output_path=os.path.join(output_dir, output_name),
            ))
        return chunks


def write_chunk_plan(path, chunks):
    '''Write chunk plan records as JSON for debugging/resume support.'''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as plan_file:
        json.dump([chunk.to_dict() for chunk in chunks], plan_file, indent=2)
