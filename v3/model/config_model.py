'''YAML configuration helpers.'''

import yaml
from collections import OrderedDict

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

