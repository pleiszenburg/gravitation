# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/loginfo.py: platform information log

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

from copy import deepcopy
from platform import (
    python_compiler,
    python_build,
    python_implementation,
    machine,
    processor,
    release,
    system,
    version,
)
import sys
from typing import Any

try:
    import cpuinfo
except ModuleNotFoundError:
    cpuinfo = None

try:
    import GPUtil
except ModuleNotFoundError:
    GPUtil = None

from .const import Threads
from .debug import typechecked

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class InfoLog:
    "holds information on the worker platform"

    def __init__(self, **meta):
        self._meta = meta

    def __getitem__(self, key: str) -> Any:
        return self._meta[key]

    def __eq__(self, other: Any) -> bool:

        self_meta = deepcopy(self.meta)
        other_meta = deepcopy(other.meta)

        for meta in (self_meta, other_meta):
            for key in ('hz_advertised_friendly', 'hz_actual_friendly', 'hz_advertised', 'hz_actual'):
                meta['cpu_info'].pop(key)
            for gpu in meta['gpu_info']:
                for key in ('load', 'memoryFree', 'memoryUsed', 'memoryUtil', 'temperature'):
                    gpu.pop(key)

        return self_meta == other_meta

    @property
    def meta(self) -> dict:
        "info meta data"

        return self._meta

    def to_dict(self) -> dict:
        "export as dict"

        return deepcopy(self._meta)

    @classmethod
    def from_dict(cls, **kwargs: Any):
        "import from dict"

        return cls(**kwargs)

    @classmethod
    def from_new(cls):
        "new info"

        return cls(
            python_build=list(python_build()),
            python_compiler=python_compiler(),
            python_implementation=python_implementation(),
            python_version=list(sys.version_info),
            os_system=system(),
            os_release=release(),
            os_version=version(),
            cpu_machine=machine(),
            cpu_processor=processor(),
            cpu_physical=Threads.physical.value,
            cpu_logical=Threads.logical.value,
            cpu_info=cpuinfo.get_cpu_info() if cpuinfo is not None else {},
            gpu_info=[
                {
                    n: getattr(gpu, n)
                    for n in dir(gpu)
                    if not n.startswith("_") and n not in ("serial", "uuid")
                }
                for gpu in GPUtil.getGPUs()
            ]
            if GPUtil is not None
            else {},
        )