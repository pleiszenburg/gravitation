# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/lib/load.py: Kernel loading infrastructure

	Copyright (C) 2019 Sebastian M. Ernst <ernst@pleiszenburg.de>

<LICENSE_BLOCK>
The contents of this file are subject to the GNU General Public License
Version 2 ("GPL" or "License"). You may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt
https://github.com/pleiszenburg/gravitation/blob/master/LICENSE

Software distributed under the License is distributed on an "AS IS" basis,
WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License for the
specific language governing rights and limitations under the License.
</LICENSE_BLOCK>

"""

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import ast
import importlib
import os
import traceback

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class _inventory(dict):
    """kernel inventory (a dict, more or less)"""

    def __init__(self):
        super().__init__()
        path = os.path.join(os.path.dirname(__file__), "..", "kernel")
        kernels = [
            (item[:-3], True) if item.lower().endswith(".py") else (item, False)
            for item in os.listdir(path)
            if not item.startswith("_")
        ]
        self.update({name: _kernel(path, name, isfile) for name, isfile in kernels})


class _kernel:
    """kernel descriptor with lazy loading of kernel module, class and meta data"""

    def __init__(self, path, name, isfile):
        self._path = path
        self._name = name
        self._isfile = isfile
        self._module = None
        self._src = None
        self._meta = None

    def __call__(self, *args, **kwargs):
        """provides access to kernel class constructor"""
        if self._module is None:
            raise SyntaxError("kernel module has not been loaded")
        return self._module.universe(*args, **kwargs)

    def __getitem__(self, key):
        """provides access to kernel meta data dict"""
        if self._meta is None:
            raise SyntaxError("kernel metadata has not been loaded")
        return self._meta[key]

    def get_class(self):
        """returns kernel class"""
        if self._module is None:
            raise SyntaxError("kernel module has not been loaded")
        return self._module.universe

    def load_meta(self):
        """loads meta data from kernel without importing it"""
        with open(
            os.path.join(self._path, self._name + ".py")
            if self._isfile
            else os.path.join(self._path, self._name, "__init__.py"),
            "r",
        ) as f:
            self._src = f.read()
        self._meta = {
            k[2:-2]: v
            for k, v in _get_vars(
                self._src,
                *[
                    "__%s__" % item
                    for item in (
                        "longname",
                        "version",
                        "description",
                        "requirements",
                        "externalrequirements",
                        "interpreters",
                        "parallel",
                        "license",
                        "authors",
                    )
                ],
            ).items()
        }
        self._meta["name"] = self._name

    def load_module(self):
        """actually imports kernel module"""
        self._module = importlib.import_module("gravitation.kernel.%s" % self._name)

    def keys(self):
        """provides access to kernel meta data dict keys"""
        if self._meta is None:
            raise SyntaxError("kernel metadata has not been loaded")
        return self._meta.keys()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def _get_vars(src, *names, default=None):
    tree = ast.parse(src)
    out_dict = {name: default for name in names}
    for item in tree.body:
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if target.id not in names:
                continue
            out_dict[target.id] = _parse_tree(item.value)
    return out_dict


def _parse_tree(leaf):
    if isinstance(leaf, ast.Str) or isinstance(leaf, ast.Bytes):
        return leaf.s
    elif isinstance(leaf, ast.Num):
        return leaf.n
    elif isinstance(leaf, ast.NameConstant):
        return leaf.value
    elif isinstance(leaf, ast.Dict):
        return {
            _parse_tree(leaf_key): _parse_tree(leaf_value)
            for leaf_key, leaf_value in zip(leaf.keys, leaf.values)
        }
    elif isinstance(leaf, ast.List):
        return [_parse_tree(leaf_item) for leaf_item in leaf.elts]
    elif isinstance(leaf, ast.Tuple):
        return tuple([_parse_tree(leaf_item) for leaf_item in leaf.elts])
    elif isinstance(leaf, ast.Set):
        return {_parse_tree(leaf_item) for leaf_item in leaf.elts}
    else:
        raise SyntaxError("unhandled type: %s (%s)" % (str(leaf), str(dir(leaf))))


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EXPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

inventory = _inventory()
