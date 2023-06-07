# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/lib/kernel.py: Kernel loading infrastructure

    Copyright (C) 2019-2023 Sebastian M. Ernst <ernst@pleiszenburg.de>

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

import importlib
import os
from typing import List, Type

from .debug import typechecked
from .errors import KernelError
from .variation import Variations

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class Kernel:
    """kernel descriptor with lazy loading of kernel meta data and class"""

    def __init__(self, name: str):
        self._name = name

        self._variations = None  # meta
        self._description = None  # meta
        self._requirements = None  # meta

        self._cls = None

    def __repr__(self) -> str:
        return f'<Kernel name={self._name:s} meta={self.meta_loaded} cls={self.cls_loaded}>'

    @property
    def meta_loaded(self) -> bool:
        return self._variations is not None

    @property
    def cls_loaded(self) -> bool:
        return self._cls is not None

    @property
    def name(self) -> str:
        """kernel name"""
        return self._name

    @property
    def variations(self) -> Variations:
        """kernel variations"""
        if not self.meta_loaded:
            raise KernelError("kernel meta data has not been loaded")
        return self._variations

    @property
    def description(self) -> str:
        """kernel description"""
        if not self.meta_loaded:
            raise KernelError("kernel meta data has not been loaded")
        return self._description

    @property
    def requirements(self) -> List[str]:
        """kernel requirements"""
        if not self.meta_loaded:
            raise KernelError("kernel meta data has not been loaded")
        return self._requirements

    @property
    def cls(self) -> Type:
        """kernel class"""
        if not self.cls_loaded:
            raise KernelError("kernel class has not been loaded")
        return self._cls

    def load_meta(self):
        """loads meta data from kernel without class or dependencies"""
        if self.meta_loaded:
            return
        module = importlib.import_module(f"gravitation.kernel.{self._name:s}")
        self._variations = module.VARIATIONS
        self._description = module.DESCRIPTION
        self._requirements = module.REQUIREMENTS

    def load_cls(self):
        """actually imports kernel class and its dependencies"""
        if self.cls_loaded:
            return
        self._cls = importlib.import_module(f"gravitation.kernel.{self._name:s}.kernel").Universe

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EXPORT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

KERNELS = {
    name: Kernel(name)
    for name in os.listdir(os.path.join(os.path.dirname(__file__), "..", "kernel"))
    if not name.startswith("_")
}
