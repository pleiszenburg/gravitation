# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/logbenchmark.py: benchmark log type

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
from shutil import get_terminal_size
import termplotlib as tpl
from typing import Any, Dict, Generator, Optional

from .debug import typechecked
from .logworker import WorkerLog
from .errors import BenchmarkLogError

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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
