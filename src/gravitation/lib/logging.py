# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/logging.py: benchmark log types

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
from io import TextIOBase
from json import decoder, dumps, loads
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
from time import time_ns
from typing import Any, List, Optional

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
from .errors import BenchmarkLogError
from .variation import Variation

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
        return self.meta == other.meta

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


@typechecked
class StepLog:
    "represents one single benchmark step"

    def __init__(self, iteration: int, runtime: int, gctime: int, runtime_min: int, gctime_min: int):
        self._iteration = iteration
        self._runtime = runtime
        self._gctime = gctime
        self._runtime_min = runtime_min
        self._gctime_min = gctime_min

    def to_dict(self) -> dict:
        "export as dict"

        return dict(
            iteration = self._iteration,
            runtime = self._runtime,
            gctime = self._gctime,
            runtime_min = self._runtime_min,
            gctime_min = self._gctime_min,
        )

    @classmethod
    def from_dict(cls, **kwargs: Any):
        "import from dict"

        return cls(**kwargs)


@typechecked
class WorkerLog:
    "represents a worker run: one length, multiple steps"

    def __init__(self,
        kernel: str,
        variation: Variation,
        info: InfoLog,
        status: str,
        length: int,
        steps: Optional[List[StepLog]] = None,
    ):
        self._kernel = kernel
        self._variation = variation
        self._info = info
        self._status = status,
        self._length = length
        self._steps = [] if steps is None else steps

    @property
    def status(self) -> str:
        "status of run: start, running, ok, traceback"

        return self._status

    @status.setter
    def status(self, value: str):
        "status of run"

        if self._status == 'ok':
            raise BenchmarkLogError('trying to change status of stopped worker run that was ok')
        if self._status not in ('start', 'running'):
            raise BenchmarkLogError('trying to change status of stopped worker run that errored')

        self._status = value

    def append(self, step: StepLog):
        "append step to worker run"

        if self._status == 'ok':
            raise BenchmarkLogError('trying to append to stopped worker run that was ok')
        if self._status not in ('start', 'running'):
            raise BenchmarkLogError('trying to append to stopped worker run that errored')

        self._steps.append(step)
        self._status = 'running'

    def to_dict(self) -> dict:
        "export as dict"

        return dict(
            kernel = self._kernel,
            variation = self._variation.to_dict(),
            info = self._info.to_dict(),
            status = self._status,
            length = self._length,
            steps = [step.to_dict() for step in self._steps],
        )

    @classmethod
    def from_dict(
        cls,
        kernel: str,
        variation: dict,
        info: dict,
        status: str,
        length: int,
        steps: List[dict],
    ):
        "import from dict"

        return cls(
            kernel = kernel,
            variation = Variation.from_dict(**variation),
            info = InfoLog.from_dict(**info),
            status = status,
            length = length,
            steps = [StepLog.from_dict(**step) for step in steps],
        )

    @classmethod
    def from_log(
        cls,
        fn: TextIOBase,
    ):
        "import from line-based log file"

        run = None

        for line in fn:

            if len(line.strip()) == 0:
                continue

            try:
                data = loads(line.rstrip('\n'))
            except decoder.JSONDecodeError as e:
                raise BenchmarkLogError('line is not valid JSON') from e

            key, value = data['key'], data['value']  # ignore time ... for now?

            if key == 'start':
                if run is not None:
                    raise BenchmarkLogError('trying to start a worker run that has been started earlier')
                run = cls.from_dict(**value)
            elif key == 'info':
                pass  # nothing to do
            elif key == 'step':
                if run is None:
                    raise BenchmarkLogError('trying to append to worker run that has not been started')
                run.append(StepLog.from_dict(**value))
            elif key == 'stop':
                if run is None:
                    raise BenchmarkLogError('trying to stop a worker run that has been stopped earlier')
                run.status = value
                return run
            else:
                raise BenchmarkLogError('unknown key')

        if run is None:
            return

        run.status = 'did not stop'
        return run

    @staticmethod
    def log(key: str, value: Any):
        "log json string to stdout"

        assert key in ('start', 'info', 'step', 'stop')
        sys.stdout.write(dumps(dict(key = key, value = value, time = time_ns())))
        sys.stdout.write("\n")
        sys.stdout.flush()
