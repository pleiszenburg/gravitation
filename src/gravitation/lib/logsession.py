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
from json import dumps
from typing import List, Optional

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

    def to_fh(self, fh: TextIOWrapper):
        "to file handle"

        fh.write(dumps([benchmark.to_dict() for benchmark in self._benchmarks], indent=4, sort_keys=True))
        fh.flush()

    def to_file(self, fn: str):
        "to file path"

        with open(fn, mode = 'w', encoding='utf-8') as f:
            self.to_fh(f)

    @classmethod
    def from_raw_fh(cls, fh: TextIOWrapper):
        "from file handle"

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
    def ingest(cls, fin: TextIOWrapper, fout: TextIOWrapper):
        "ingest raw worker log file and convert to session log"

        session = cls.from_raw_fh(fin)
        session.to_fh(fout)
