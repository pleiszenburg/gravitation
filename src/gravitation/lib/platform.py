# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/platform.py: platform information log

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
from json import dumps, loads
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
from typing import Any, Dict, Generator, Union, Tuple

import psutil

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
from .option import Option

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class Platform:
    "holds information on the worker platform - IMMUTABLE"

    _current = None

    def __init__(self, **meta: Union[str, int]):
        self._meta = meta

    def __getitem__(self, key: str) -> Union[str, int]:
        return self._meta[key]

    def __repr__(self) -> str:
        return f'<Platform {self._meta["python_implementation"]:s} {self._meta["python_version"]:s} {self._meta["cpu_machine"]:s} {self._meta["os_system"]:s} {self._meta["os_release"]:s} {self._meta["cpu_info"]:s} ({self._meta["cpu_ram"]:d}G, {len(self._meta["gpu_info"].split(";"))} gpu[s])>'

    def __eq__(self, other: Any) -> bool:

        if not isinstance(other, type(self)):
            return NotImplemented

        return self.key == other.key

    @property
    def key(self) -> Tuple[str, ...]:
        "unique immutable key for set-like operations"

        return tuple(self[field] for field in self.keys())

    def keys(self) -> Generator:
        "names of fields in meta data"

        yield from sorted(self._meta.keys())

    def to_json(self) -> str:
        "export as json"

        return dumps(self.to_dict(), sort_keys = True)

    def to_dict(self) -> Dict[str, Union[str, int]]:
        "export as dict"

        return deepcopy(self._meta)

    def to_options(self) -> Tuple[Option, ...]:
        "export to tuples that can be used to generate click command options"

        return tuple(Option(k, v) for k, v in self._meta)

    @staticmethod
    def get_cpu() -> str:
        "get information on cpu"

        if cpuinfo is None:
            return "[cpuinfo not present]"

        return cpuinfo.get_cpu_info()['brand_raw']

    @staticmethod
    def get_gpus() -> str:
        "get information on gpus"

        if GPUtil is None:
            return "[GPUtil not present]"

        gpus = [
            {
                n: getattr(gpu, n)
                for n in dir(gpu)
                if not n.startswith("_") and n not in ("serial", "uuid")
            }
            for gpu in GPUtil.getGPUs()
        ]
        gpus = [
            f'{gpu["name"]} ({round(gpu["memoryTotal"]/1024)}G, driver={gpu["driver"]} display_active={gpu["display_active"]}, display_mode={gpu["display_mode"]})'
            for gpu in gpus
        ]
        return '; '.join(gpus)

    @classmethod
    def from_dict(cls, **kwargs: Union[str, int]):
        "import from dict"

        return cls(**kwargs)

    @classmethod
    def from_json(cls, raw: str):
        "import from json"

        return cls.from_dict(**loads(raw))

    @classmethod
    def from_current(cls):
        "current platform"

        if cls._current is None:
            cls._current = cls(
                python_build=', '.join(python_build()),
                python_compiler=python_compiler(),
                python_implementation=python_implementation(),
                python_version=f'{sys.version_info.major:d}.{sys.version_info.minor:d}.{sys.version_info.micro:d}-{sys.version_info.releaselevel:s}-{sys.version_info.serial:d}',
                os_system=system(),
                os_release=release(),
                os_version=version(),
                cpu_machine=machine(),
                cpu_processor=processor(),
                cpu_physical=Threads.physical.value,
                cpu_logical=Threads.logical.value,
                cpu_ram=psutil.virtual_memory().total // 1024**3,
                cpu_info=cls.get_cpu(),
                gpu_info=cls.get_gpus(),
            )

        return cls._current
