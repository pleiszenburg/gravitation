# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/logworker.py: worker log type

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

from io import TextIOWrapper
from json import decoder, dumps, loads
import sys
from time import time_ns
from typing import Any, Dict, Generator, List, Optional

from .debug import typechecked
from .errors import BenchmarkLogError
from .loginfo import InfoLog
from .logstep import StepLog
from .variation import Variation

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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
        return (self._steps[iteration] for iteration in self.iterations())

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

        return self._steps[self.iterations()[-1]].runtime_min

    @property
    def gctime_min(self) -> int:
        "minimal gc time of worker iteration"

        if len(self) == 0:
            raise BenchmarkLogError('no data available')

        return self._steps[self.iterations()[-1]].gctime_min

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

    def iterations(self) -> List[int]:
        "sorted generator of available iterations"

        return sorted(self._steps.keys())

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
