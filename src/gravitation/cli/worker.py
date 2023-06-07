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

try:
    import cpuinfo
except ModuleNotFoundError:
    cpuinfo = None

try:
    import GPUtil
except ModuleNotFoundError:
    GPUtil = None

from ._kernel import add_kernel_commands
from ..lib.baseuniverse import BaseUniverse
from ..lib.const import DEFAULT_LEN
from ..lib.debug import typechecked
from ..lib.errors import VariationError
from ..lib.kernel import Kernel, KERNELS
from ..lib.timing import BestRunTimer, ElapsedTimer
from ..lib.variation import Variation

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@click.group(short_help="isolated single-kernel benchmark worker")
@click.option(
    "--len",
    default=DEFAULT_LEN,
    type=int,
    show_default=True,
    help="number of point masses",
)
@click.option(
    "--datafile",
    "-d",
    default="data.h5",
    type=str,
    show_default=True,
    help="name data file",
)
@click.option(
    "--save_after_iteration",
    "-s",
    type=int,
    show_default=True,
    multiple=True,
    help="save model universe into file iteration n",
)
@click.option(
    "--read_initial_state",
    "-r",
    default=False,
    is_flag=True,
    show_default=True,
    help="read initial state from data file",
)
@click.option(
    "--min_iterations",
    "-i",
    default=10,
    type=int,
    show_default=True,
    help="minimum number of simulation steps; if save_after_iteration is specified and larger, it takes precedence and runs at least this many iterations",
)
@click.option(
    "--min_total_runtime",
    "-t",
    default=10,
    type=int,
    show_default=True,
    help="minimal total runtime of (all) steps, in seconds",
)
@click.pass_context
def worker(
    ctx,
    len,
    datafile,
    save_after_iteration,
    read_initial_state,
    min_iterations,
    min_total_runtime,
):
    "isolated single-kernel benchmark worker entry point"

    def run(kernel: Kernel, **kwargs):

        kernel.load_meta()
        try:
            kernel.variations.select(**kwargs)
        except VariationError as e:
            kernel.variations.print()
            print('ERROR:', e)
            sys.exit(1)

        _Worker(
            kernel = kernel.name,
            variation = kernel.variations.selected,
            length = len,
            datafile = datafile,
            save_after_iteration = save_after_iteration,
            read_initial_state = read_initial_state,
            min_iterations = min_iterations,
            min_total_runtime = min_total_runtime,
        ).run()

    ctx.meta['run'] = run  # runs via kernel sub-command


add_kernel_commands(worker)


@typechecked
class _Worker:
    "isolated single-kernel benchmark worker class"

    def __init__(
        self,
        kernel: str,
        variation: Variation,
        length: int,
        datafile: str,
        save_after_iteration: Tuple[int, ...],
        read_initial_state: bool,
        min_iterations: int,
        min_total_runtime: int,
    ):
        self._msg(log="START")

        self._kernel = kernel
        self._length = length
        self._datafile = datafile
        self._save_after_iteration = save_after_iteration
        self._read_initial_state = read_initial_state
        self._min_iterations = min_iterations
        if (
            len(self._save_after_iteration) > 0
            and max(self._save_after_iteration) > self._min_iterations
        ):
            self._min_iterations = max(self._save_after_iteration)
        self._min_total_runtime = min_total_runtime * 10**9  # convert to ns
        self._variation = variation

        self._counter = 0
        self._rt = BestRunTimer()  # runtime
        self._gt = BestRunTimer()  # gc time
        self._et = None  # elapsed time, set up later

        self._msg_inputs()
        self._universe = self._init_universe()

    def _init_universe(self) -> BaseUniverse:
        self._msg(log="PROCEDURE", msg="Creating simulation ...")

        KERNELS[self._kernel].load_cls()

        try:
            if self._read_initial_state:
                universe = (
                    KERNELS[self._kernel].cls.from_hdf5(
                        fn=self._datafile,
                        gn=KERNELS[self._kernel].cls.export_name_group(kernel = 'zero', length = self._length, steps = 0),
                        variation=self._variation,
                    )
                )
                assert self._length == len(universe)
            else:
                universe = (
                    KERNELS[self._kernel].cls.from_galaxy(
                        length=self._length,
                        variation=self._variation,
                    )
                )
        except Exception:
            self._exit(ok = False)

        self._msg(log="PROCEDURE", msg="Simulation created.")
        self._msg(log="SIZE", value=len(universe))

        return universe

    @staticmethod
    def _msg(**d):
        sys.stdout.write(f"{json.dumps(d):s}\n")
        sys.stdout.flush()

    def _msg_inputs(self):
        self._msg(
            log="INPUT",
            kernel=dict(
                name=self._kernel,
                **self._variation.to_dict(),
            ),
            simulation=dict(
                length=self._length,
                min_iterations=self._min_iterations,
                min_total_runtime=self._min_total_runtime,
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
                physical_cores=psutil.cpu_count(logical=False),
                logical_cores=psutil.cpu_count(logical=True),
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
            self._universe.push_stage1()
            gc.collect()
            self._universe.gc_collect()
            self._rt.start()
            self._universe.step_stage1()
            rt_ = self._rt.stop()
            self._gt.start()
            gc.collect()
            gt_ = self._gt.stop()
        except Exception:
            self._exit(ok = False)

        try:
            self._universe.step(stage1=False)
            self._universe.assert_not_isnan()
        except Exception:
            self._exit(ok = False)

        self._counter += 1
        if self._counter in self._save_after_iteration:
            self._store()

        self._msg(log="STEP", runtime=rt_, gctime=gt_, counter=self._counter)
        self._msg(log="BEST_TIME", value=self._rt.min)

    def _store(self):
        self._msg(log="PROCEDURE", msg=f"Saving data after step {self._counter:d} ...")

        try:
            self._universe.to_hdf5(
                fn=self._datafile,
                gn=self._universe.export_name_group(
                    kernel=self._kernel,
                    len=len(self._universe),
                    step=self._counter,
                    **self._variation.from_dict(),
                )
            )
        except Exception:
            self._exit(ok = False)

        self._msg(log="PROCEDURE", msg=f"Data saved after step {self._counter:d}.")

    def _exit(self, ok: bool = True):
        if hasattr(self, "_universe"):
            self._universe.stop()
        if not ok:
            self._msg(log="ERROR", msg=traceback.format_exc())
        self._msg(log="EXIT", msg="OK" if ok else "BAD")
        sys.exit(int(not ok))

    def run(self):
        "run worker"

        self._universe.start()

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
            self._exit()
            return

        self._msg(log="PROCEDURE", msg="Extra steps required.")
        time_remaining = self._min_total_runtime - et_
        iterations_remaining = time_remaining // et_ * self._min_iterations

        # required extra runs until min_total_runtime
        for _ in range(iterations_remaining):
            self._step()

        self._msg(log="PROCEDURE", msg="Extra steps finished.")
        self._exit()

@typechecked
def worker_command(
    datafile: str,
    kernel: str,
    length: int,
    save_after_iteration: Tuple[int, ...],
    read_initial_state: bool,
    min_iterations: int,
    min_total_runtime: int,
    **kwargs,  # variation
) -> List[str]:
    "returns command list for use with subprocess.Popen"

    cmd = [
        "gravitation",
        "worker",
        "--len",
        f"{length:d}",
        "--datafile",
        datafile,
        *list(
            chain(
                *[("--save_after_iteration", f"{it:d}") for it in save_after_iteration]
            )
        ),
        "--min_iterations",
        f"{min_iterations:d}",
        "--min_total_runtime",
        f"{min_total_runtime:d}",
    ]

    if read_initial_state:
        cmd.append("--read_initial_state")

    cmd.append(kernel)
    for key, value in kwargs.items():
        cmd.extend((f'--{key:s}', str(value)))

    return cmd
