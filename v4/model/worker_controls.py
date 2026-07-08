'''Shared worker-limit resolution helpers for the v4 pipeline.'''

import os


def get_slurm_cpu_cap():
    '''Return SLURM_CPUS_PER_TASK as an integer ceiling when available.'''
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if not slurm_cpus:
        return None
    try:
        return max(1, int(slurm_cpus))
    except ValueError:
        return None


def resolve_worker_limits(configured_workers, num_required_chunks):
    '''Resolve an effective worker count with SLURM and chunk ceilings.'''
    chunk_cap = max(1, int(num_required_chunks))
    slurm_cpu_cap = get_slurm_cpu_cap()
    fallback_workers = slurm_cpu_cap or os.cpu_count() or 1
    requested_workers = configured_workers if configured_workers else fallback_workers

    ceilings = [requested_workers, chunk_cap]
    if slurm_cpu_cap is not None:
        ceilings.append(slurm_cpu_cap)

    effective_workers = max(1, min(ceilings))
    return {
        'configured_workers': configured_workers,
        'requested_workers': requested_workers,
        'effective_workers': effective_workers,
        'chunk_cap': chunk_cap,
        'slurm_cpu_cap': slurm_cpu_cap,
        'fallback_workers': fallback_workers,
    }
