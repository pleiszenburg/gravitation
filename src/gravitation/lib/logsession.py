# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/logsession.py: session log type

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
from json import dumps, loads
from typing import Any, Generator, List, Optional

from .debug import typechecked
from .logbenchmark import BenchmarkLog
from .logworker import WorkerLog

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class SessionLog:
    "handle session, i.e. group of benchmarks"

    def __init__(self, benchmarks: Optional[List[BenchmarkLog]] = None):
        self._benchmarks = [] if benchmarks is None else benchmarks

    def __len__(self) -> int:
        return len(self._benchmarks)

    def __repr__(self) -> str:
        return f'<SessionLog len={len(self):d}>'

    def __getitem__(self, idx: int) -> BenchmarkLog:
        return self._benchmarks[idx]

    def __iter__(self) -> Generator:
        return (benchmark for benchmark in self._benchmarks)

    def merge(self, other: Any):
        "merge other session into this one"

        if not isinstance(other, type(self)):
            raise TypeError()

        self._benchmarks.extend(other)

    def to_runtime_figure(self):
        "for use with plotting tools"

        traces = [
            benchmark.to_runtime_trace()
            for benchmark in self
        ]

        xs = sorted({
            value
            for trace in traces
            for value in trace['x']
        })
        ymin = min({
            value
            for trace in traces
            for value in trace['y']
        })
        for idx in range(2, 4):
            traces.append(dict(
                type="scatter",
                x=xs.copy(),
                y=[
                    (ymin / (xs[0] ** idx)) * (x ** idx)
                    for x in xs
                ],
                name=f'x^{idx:d}',
                mode="lines",
                line_color='rgba(200,200,200,128)',
            ))

        layout = dict(
            autosize=True,
            xaxis=dict(
                type="log",
                autorange=True,
                title="items per simulation",
            ),
            yaxis=dict(
                type="log",
                autorange=True,
                scaleanchor="x",
                scaleratio=0.3,
                title="time per iteration [s]",
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode="x unified",
        )

        return dict(
            data = traces,
            layout = layout,
        )

    def to_dict(self) -> dict:
        "export as dict"

        return dict(
            benchmarks = [benchmark.to_dict() for benchmark in self._benchmarks]
        )

    def to_fh(self, fh: TextIOWrapper):
        "to file handle"

        fh.write(dumps(self.to_dict(), indent=4, sort_keys=True))
        fh.flush()

    def to_file(self, fn: str):
        "to file path"

        with open(fn, mode = 'w', encoding='utf-8') as f:
            self.to_fh(f)

    @classmethod
    def from_dict(cls, benchmarks: dict):
        "import from dict"

        return cls(benchmarks = [
            BenchmarkLog.from_dict(**benchmark)
            for benchmark in benchmarks
        ])

    @classmethod
    def from_fh(cls, fh: TextIOWrapper):
        "from file handle"

        return cls.from_dict(**loads(fh.read()))

    @classmethod
    def from_file(cls, fn: str):
        "from file"

        with open(fn, mode = 'r', encoding = 'utf-8') as f:
            return cls.from_fh(f)

    @classmethod
    def from_log_fh(cls, fh: TextIOWrapper):
        "from log file handle"

        workers = []

        while True:
            worker = WorkerLog.from_fh(fh)
            if worker is None:
                break
            workers.append(worker)

        benchmarks = []
        for worker in workers:
            found = False
            for benchmark in benchmarks:
                if not benchmark.matches(worker):
                    continue
                benchmark.add(worker)
                found = True
                break
            if found:
                continue
            benchmarks.append(BenchmarkLog({worker.length: worker}))

        return cls(benchmarks)

    @classmethod
    def from_log_file(cls, fn: str):
        "from log file"

        with open(fn, mode = 'r', encoding = 'utf-8') as f:
            return cls.from_log_fh(f)
