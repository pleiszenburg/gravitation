# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/lib/benchmark.py: benchmark wrapper

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
from typing import Generator, List

from .baseuniverse import BaseUniverse
from .const import Display, Stream
from .debug import typechecked
from .errors import BenchmarkLogError
from .kernel import KERNELS
from .logbenchmark import BenchmarkLog
from .logworker import WorkerLog
from .proc import run_command
from .worker import Worker

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class Benchmark:
    "running benchmarks and handling worker logs"

    def __init__(
        self,
        fh: TextIOWrapper,
        display: Display,
    ):

        self._fh = fh
        self._display = display

        self._log = BenchmarkLog()

        self._error = None

    def __call__(
        self,
        stream_id: Stream,
        line: str,
    ):

        if stream_id is Stream.stderr:
            line = WorkerLog.encode(key = "stderr", value = line)

        self._fh.write(f"{line:s}\n")
        self._fh.flush()

        if self._display is Display.log:
            print(line)
        if stream_id is Stream.stderr:
            return

        if self._error is None:
            try:
                msg = WorkerLog.decode(line)
            except BenchmarkLogError as e:
                if self._display is not Display.log:  # has already been printed
                    print(line)
                self._error = e
                return
        else:
            if self._display is not Display.log:  # has already been printed
                print(line)
            return

        self._log.live(**msg)

        if self._display is not Display.plot or len(self._log) == 0:
            return

        self._log.plot_cli()

    def join(self):
        "terminate processing"

        if self._error is None:
            return

        raise self._error

    @staticmethod
    def sq_range(start: int, stop: int) -> Generator:
        """special range generator, going from 2^start to 2^stop with some interpolation"""

        if start > stop:
            raise ValueError()

        for value in range(start, stop):
            yield 2 ** value
            yield round(2 ** (value + 0.5))
        yield 2 ** stop

    @classmethod
    def common_initial_states(cls, start: int, stop: int, datafile: int):
        """create common initial states for benchmarks for later evaluation of results"""

        class UniverseZero(BaseUniverse):
            "Generating common start universe"

            def step_stage1(self):
                "not required here"

        for length in cls.sq_range(start, stop):
            print(
                f"Creating initial state for {length:d} masses (max {2**stop:d}) ..."
            )
            initial_state = UniverseZero.from_galaxy(length=length)
            initial_state.to_hdf5(fn=datafile, gn=UniverseZero.export_name_group(kernel = "zero", length = length, steps = 0))

    @classmethod
    def run(
        cls,
        logfile: str,
        datafile: str,
        common_initial_state: bool,
        kernels: List[str],
        sq_range_start: int,
        sq_range_stop: int,
        save_after_iteration: List[int],
        min_iterations: int,
        min_total_runtime: int,
        display: Display,
    ):
        "run set of benchmarks via workers"

        if common_initial_state:
            cls.common_initial_states(sq_range_start, sq_range_stop, datafile)

        with open(logfile, "w", encoding="utf-8") as fh:

            for kernel in kernels:

                KERNELS[kernel].load_meta()

                for variation in KERNELS[kernel].variations:

                    if variation['threads'].name.startswith('t'):
                        continue

                    session = cls(
                        fh=fh,
                        display=display,
                    )

                    for length in cls.sq_range(sq_range_start, sq_range_stop):

                        run_command(
                            Worker.command(
                                datafile=datafile,
                                kernel=kernel,
                                length=length,
                                save_after_iteration=save_after_iteration,
                                read_initial_state=common_initial_state,
                                min_iterations=min_iterations,
                                min_total_runtime=min_total_runtime,
                                **variation.to_dict(),
                            ),
                            unbuffer=True,
                            processing=session,
                        )
