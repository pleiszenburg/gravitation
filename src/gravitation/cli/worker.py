# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/worker.py: worker command

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

import gc
from itertools import chain
import json
import platform
import sys
import traceback
from typing import List, Tuple

import click
import psutil
from typeguard import typechecked

try:
    import cpuinfo
except ModuleNotFoundError:
    cpuinfo = None

try:
    import GPUtil
except ModuleNotFoundError:
    GPUtil = None

from ..kernel._base import UniverseBase
from ..lib.load import inventory
from ..lib.timing import BestRunTimer, ElapsedTimer

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONST
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

MAX_TREADS = psutil.cpu_count(logical=True)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@click.command(short_help="isolated single-kernel benchmark worker")
@click.option(
    "--kernel",
    "-k",
    type=click.Choice(sorted(list(inventory.keys()))),
    required=True,
    help="name of kernel module",
)
@click.option(
    "--len",
    default=2000,
    type=int,
    show_default=True,
    help="number of point masses",
)
@click.option(
    "--data_out_file",
    "-o",
    default="data.h5",
    type=str,
    show_default=True,
    help="name of output data file",
)
@click.option(
    "--save_after_iteration",
    "-s",
    type=int,
    show_default=True,
    multiple=True,
    help="save model universe into file iteration x, -1 if nothing should be saved",
)
@click.option(
    "--min_iterations",
    "-i",
    default=10,
    type=int,
    show_default=True,
    help="minimum number of simulation steps",
)
@click.option(
    "--min_total_runtime",
    "-t",
    default=10,
    type=int,
    show_default=True,
    help="minimal total runtime of (all) steps, in seconds",
)
@click.option(
    "--threads",
    "-p",
    default="1",
    type=click.Choice([str(i) for i in range(1, MAX_TREADS + 1)]),
    show_default=True,
    help="number of threads/processes for parallel implementations",
)
def worker(
    kernel,
    len,
    data_out_file,
    save_after_iteration,
    min_iterations,
    min_total_runtime,
    threads,
):
    "isolated single-kernel benchmark worker entry point"

    _Worker(
        kernel,
        len,
        data_out_file,
        save_after_iteration,
        min_iterations,
        min_total_runtime,
        int(threads),
    ).run()


@typechecked
class _Worker:
    "isolated single-kernel benchmark worker class"

    def __init__(
        self,
        kernel: str,
        len: int,
        data_out_file: str,
        save_after_iteration: Tuple[int, ...],
        min_iterations: int,
        min_total_runtime: int,
        threads: int,
    ):

        self._msg(log="START")

        self._kernel = kernel
        self._len = len
        self._data_out_file = data_out_file
        self._save_after_iteration = save_after_iteration
        self._min_iterations = min_iterations
        self._min_total_runtime = min_total_runtime * 10**9  # convert to ns
        self._threads = threads

        self._counter = 0
        self._rt = BestRunTimer()  # runtime
        self._gt = BestRunTimer()  # gc time
        self._et = None  # elapsed time, set up later

        self._msg_inputs()
        self._universe = self._init_universe()

    def _init_universe(self) -> UniverseBase:

        self._msg(log="PROCEDURE", msg="Creating simulation ...")

        inventory[self._kernel].load_module()

        try:
            universe = inventory[self._kernel].get_class().from_galaxy(
                stars_len = self._len,
                threads = self._threads,
            )
        except Exception:
            self._msg(log="ERROR", msg=traceback.format_exc())
            self._msg(log="EXIT", msg="BAD")
            sys.exit()

        self._msg(log="PROCEDURE", msg="Simulation created.")
        self._msg(log="SIZE", value=len(universe))

        return universe

    @staticmethod
    def _msg(**d):

        sys.stdout.write(f'{json.dumps(d):s}\n')
        sys.stdout.flush()

    def _msg_inputs(self):

        self._msg(
            log="INPUT",
            simulation=dict(
                kernel=self._kernel,
                stars_len=self._len,
                min_iterations=self._min_iterations,
                min_total_runtime=self._min_total_runtime,
                threads=self._threads,
            ),
            python=dict(
                build=list(platform.python_build()),
                compiler=platform.python_compiler(),
                implementation=platform.python_implementation(),
                version=list(sys.version_info),
            ),
            platform=dict(
                system=platform.system(),
                release=platform.release(),
                version=platform.version(),
                machine=platform.machine(),
                processor=platform.processor(),
                cores=psutil.cpu_count(logical=False),
                threads=MAX_TREADS,
                _cpu=cpuinfo.get_cpu_info() if cpuinfo is not None else {},
                _gpu=[
                    {
                        n: getattr(gpu, n)
                        for n in dir(gpu)
                        if not n.startswith("_") and n not in ("serial", "uuid")
                    }
                    for gpu in GPUtil.getGPUs()
                ]
                if GPUtil is not None
                else {},
            ),
        )

    def _step(self):

        try:
            gc.collect()
            self._rt.start()
            self._universe.step_stage1()
            rt_ = self._rt.stop()
            self._gt.start()
            gc.collect()
            gt_ = self._gt.stop()
        except Exception:
            self._msg(log="ERROR", msg=traceback.format_exc())
            self._msg(log="EXIT", msg="BAD")
            sys.exit()

        try:
            self._universe.step(stage1 = False)
        except Exception:
            self._msg(log="ERROR", msg=traceback.format_exc())
            self._msg(log="EXIT", msg="BAD")
            sys.exit()

        self._counter += 1
        if self._counter in self._save_after_iteration:
            self._store()

        self._msg(log="STEP", runtime=rt_, gctime=gt_, counter=self._counter)
        self._msg(log="BEST_TIME", value=self._rt.min)

    def _store(self):

        self._msg(log="PROCEDURE", msg=f"Saving data after step {self._counter:d} ...")

        try:
            self._universe.to_hdf5(
                fn = self._data_out_file,
                gn = f"kernel={self._kernel:s};len={len(self._universe):d};step={self._counter:d}",
            )
        except Exception:
            self._msg(log="ERROR", msg=traceback.format_exc())
            self._msg(log="EXIT", msg="BAD")
            sys.exit()

        self._msg(log="PROCEDURE", msg=f"Data saved after step {self._counter:d}.")

    def run(self):
        "run worker"

        gc.disable()

        if 0 in self._save_after_iteration:
            self._store()

        self._et = ElapsedTimer()  # elapsed time

        # required min runs
        for _ in range(self._min_iterations):
            self._step()

        # does elapsed time satisfy min_total_runtime?
        et_ = self._et()
        if et_ >= self._min_total_runtime:
            self._msg(log="PROCEDURE", msg="Minimum steps sufficient.")
            self._msg(log="EXIT", msg="OK")
            sys.exit()

        self._msg(log="PROCEDURE", msg="Extra steps required.")
        time_remaining = self._min_total_runtime - et_
        iterations_remaining = time_remaining // et_ * self._min_iterations

        # required extra runs until min_total_runtime
        for _ in range(iterations_remaining):
            self._step()

        self._msg(log="EXIT", msg="OK")
        sys.exit()


@typechecked
def worker_command(
    data_out_file: str,
    interpreter: str,
    kernel: str,
    len: int,
    save_after_iteration: Tuple[int, ...],
    min_iterations: int,
    min_total_runtime: int,
    threads: int,
) -> List[str]:
    "returns command list for use with subprocess.Popen"

    return [
        interpreter,
        "-c",
        "from gravitation.cli import cli; cli()",
        "worker",
        "--kernel",
        kernel,
        "--len",
        f"{len:d}",
        "--data_out_file",
        data_out_file,
        *list(
            chain(
                *[("--save_after_iteration", f"{it:d}") for it in save_after_iteration]
            )
        ),
        "--min_iterations",
        f"{min_iterations:d}",
        "--min_total_runtime",
        f"{min_total_runtime:d}",
        "--threads",
        f"{threads:d}",
    ]
