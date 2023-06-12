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
from io import TextIOWrapper
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
from shutil import get_terminal_size
import sys
import termplotlib as tpl
from time import time_ns
from typing import Any, Dict, Generator, Optional

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


@typechecked
class StepLog:
    "represents one single benchmark step"

    def __init__(self, iteration: int, runtime: int, gctime: int, runtime_min: int, gctime_min: int):
        self._iteration = iteration
        self._runtime = runtime
        self._gctime = gctime
        self._runtime_min = runtime_min
        self._gctime_min = gctime_min

    @property
    def iteration(self) -> int:
        "iteration of step"

        return self._iteration

    @property
    def runtime(self) -> int:
        "runtime of step"

        return self._runtime

    @property
    def gctime(self) -> int:
        "gc time of step"

        return self._iteration

    @property
    def runtime_min(self) -> int:
        "current minimal runtime"

        return self._runtime_min

    @property
    def gctime_min(self) -> int:
        "current minimal gc time"

        return self._gctime_min

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
        steps: Optional[Dict[int, StepLog]] = None,
    ):
        self._kernel = kernel
        self._variation = variation
        self._info = info
        self._status = status
        self._length = length
        self._steps = {} if steps is None else steps

    def __len__(self) -> int:
        return len(self._steps)

    def __getitem__(self, iteration: int) -> StepLog:
        try:
            return self._steps[iteration]
        except KeyError as e:
            raise BenchmarkLogError('iteration not present in log') from e

    def __iter__(self) -> Generator:
        return (self._steps[iteration] for iteration in sorted(self.iterations()))

    @property
    def kernel(self) -> str:
        "kernel"

        return self._kernel

    @property
    def variation(self) -> Variation:
        "kernel variation"

        return self._variation

    @property
    def info(self) -> InfoLog:
        "info"

        return self._info

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

    @property
    def length(self) -> int:
        "length of simulation"

        return self._length

    @property
    def runtime_min(self) -> int:
        "minimal runtime of worker iteration"

        if len(self) == 0:
            raise BenchmarkLogError('no data available')

        return self._steps[list(self.iterations())[-1]].runtime_min

    @property
    def gctime_min(self) -> int:
        "minimal gc time of worker iteration"

        if len(self) == 0:
            raise BenchmarkLogError('no data available')

        return self._steps[list(self.iterations())[-1]].gctime_min

    def add(self, step: StepLog):
        "add step to worker run"

        if self._status == 'ok':
            raise BenchmarkLogError('trying to add to stopped worker run that was ok')
        if self._status not in ('start', 'running'):
            raise BenchmarkLogError('trying to add to stopped worker run that errored')
        if step.iteration in self.iterations():
            raise BenchmarkLogError('trying to add step with iteration that is already present')

        self._steps[step.iteration] = step
        self._status = 'running'

    def iterations(self) -> Generator:
        "sorted generator of available iterations"

        return (iteration for iteration in sorted(self._steps.keys()))

    def live(self, key: str, value: Any, time: int):
        "handle incoming live stream of logs"

        if key == 'start':
            raise BenchmarkLogError('trying to start a worker run that has been started earlier')

        if key == 'info':
            return  # nothing to do

        if key == 'step':
            self.add(StepLog.from_dict(**value))
            return

        if key == 'stop':
            if self._status not in ('start', 'running'):
                raise BenchmarkLogError('trying to stop a worker run that has been stopped earlier')
            self._status = value
            return

        raise BenchmarkLogError('unknown key')


    def matches(self, other: Any) -> bool:
        "check if workers belong to same benchmark"

        assert isinstance(other, type(self))

        return all((
            self.kernel == other.kernel,
            self.variation == other.variation,
            self.info == other.info,
        ))

    def to_dict(self) -> dict:
        "export as dict"

        return dict(
            kernel = self._kernel,
            variation = self._variation.to_dict(),
            info = self._info.to_dict(),
            status = self._status,
            length = self._length,
            steps = {step.iteration: step.to_dict() for step in self._steps.items()},
        )

    @classmethod
    def from_dict(
        cls,
        kernel: str,
        variation: dict,
        info: dict,
        status: str,
        length: int,
        steps: Dict[int, dict],
    ):
        "import from dict"

        return cls(
            kernel = kernel,
            variation = Variation.from_dict(**variation),
            info = InfoLog.from_dict(**info),
            status = status,
            length = length,
            steps = {iteration: StepLog.from_dict(**step) for iteration, step in steps.items()},
        )

    @classmethod
    def from_fh(
        cls,
        fn: TextIOWrapper,
    ):  # -> Self or None
        "import from line-based decoded file or stream via handle"

        run = None

        for line in fn:

            if len(line.strip()) == 0:
                continue

            data = cls.decode(line)
            key, value = data['key'], data['value']  # ignore time ... for now?

            if key == 'start':
                if run is not None:
                    raise BenchmarkLogError('trying to start a worker run that has been started earlier')
                run = cls.from_dict(**value)
            elif key == 'info':
                pass  # nothing to do
            elif key == 'step':
                if run is None:
                    raise BenchmarkLogError('trying to add to worker run that has not been started')
                run.add(StepLog.from_dict(**value))
            elif key == 'stop':
                if run is None:
                    raise BenchmarkLogError('trying to stop a worker run that has been stopped earlier')
                run.status = value
                return run
            else:
                raise BenchmarkLogError('unknown key')

        if run is not None:
            run.status = 'did not stop'

        return run  # Log object or None

    @classmethod
    def log(cls, key: str, value: Any):
        "log json string to stdout"

        assert key in ('start', 'info', 'step', 'stop')
        sys.stdout.write(f'{cls.encode(key = key, value = value):s}\n')
        sys.stdout.flush()

    @staticmethod
    def encode(key: str, value: Any) -> str:
        "encode log package to line string"

        try:
            return dumps(dict(key = key, value = value, time = time_ns()))
        except Exception as e:  # likely type error
            raise BenchmarkLogError('can not be converted to valid JSON') from e

    @staticmethod
    def decode(line: str) -> dict:
        "decode log package from line string"

        try:
            return loads(line.rstrip('\n'))
        except decoder.JSONDecodeError as e:
            raise BenchmarkLogError('line is not valid JSON') from e


@typechecked
class BenchmarkLog:
    "represents multiple worker runs, multiple lengths, one kernel & variation"

    def __init__(self, workers: Optional[Dict[int, WorkerLog]] = None):

        if workers is not None and len(workers) > 1:
            verify = next(map(lambda x: x, workers.values()))
            if any(verify != worker for worker in workers.values):
                raise BenchmarkLogError('workers inconsistent')

        self._workers = {} if workers is None else workers
        self._current = None

    def __len__(self) -> int:
        return len(self._workers)

    def __getitem__(self, length: int) -> WorkerLog:
        try:
            return self._workers[length]
        except KeyError as e:
            raise BenchmarkLogError('length not present in benchmark') from e

    def __iter__(self) -> Generator:
        return (self._workers[length] for length in sorted(self._workers.keys()))

    def _get(self):
        "return random worker if present or None"

        if len(self) == 0:
            return None

        return next(map(lambda x: x, self._workers.values()))

    def add(self, worker: WorkerLog):
        "add worker to benchmark run"

        if not self.matches(worker):
            raise BenchmarkLogError('worker does not belong to benchmark')
        if worker.length in self._workers.keys():
            raise BenchmarkLogError('length already present in benchmark')

        self._workers[worker.length] = worker

    def lengths(self) -> Generator:
        "sorted generator of available lengths"

        return (length for length in self._workers.keys())

    def live(self, key: str, value: Any, time: int):
        "handle incoming live stream of logs"

        if key == "start":
            self._current = WorkerLog.from_dict(**value)
            self.add(self._current)
            return
        elif key != 'start' and (
            self._current is None or self._current.status not in ('start', 'running')
        ):
            return  # TODO some kind of error ocurred

        self._current.live(key = key, value = value, time = time)

    def matches(self, worker: WorkerLog) -> bool:
        "check if workers belong to same benchmark"

        if len(self._workers) == 0:
            return True

        return self._get().matches(worker)

    def plot_cli(self):
        "plot current state of benchmark"

        if len(self) == 0:
            return

        data = {
            length: self._workers[length].runtime_min * 1e-9
            for length in self.lengths()
            if len(self._workers[length]) > 0
        }

        if len(data) == 0:
            return

        x = sorted(data.keys())  # TODO property
        y = [data[length] for length in x]  # TODO property

        current_length = x[-1]
        current_iteration = list(self._workers[current_length].iterations())[-1]
        t = get_terminal_size((80, 20))
        label = " / ".join([
            f"kernel={self._get().kernel:s}",
            f"variation={repr(self._get().variation):s}",
            f"len={current_length:d}",
            f"iteration={current_iteration:d}",
            f"best={y[-1]:.02e}s",
        ])

        fig = tpl.figure()
        fig.plot(
            x,
            y,
            label=label,
            width=t.columns,
            height=t.lines,
            extra_gnuplot_arguments=[
                "set logscale x 2",
                'set format y "10^{%L}"',
                "set logscale y 10",
            ],
        )
        fig.show()

    @classmethod
    def from_fh(
        cls,
        fn: TextIOWrapper,
    ):  # -> List[Self]
        "import from line-based decoded file or stream via handle"

        # TODO
