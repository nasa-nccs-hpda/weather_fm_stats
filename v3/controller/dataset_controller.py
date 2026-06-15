'''Dataset workflow controller helpers.

This module is intentionally light in the first v3 pass. Dataset execution is
still coordinated by cli_controller while the model layer is being separated.
'''

from model.dataset_processor import BatchDatasetProcessor

__all__ = ['BatchDatasetProcessor']
