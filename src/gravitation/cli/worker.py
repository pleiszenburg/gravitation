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
import itertools
import json
import platform
import sys
import traceback

import click
import psutil

try:
    import cpuinfo

    CPUINFO = True
except:
    CPUINFO = False

try:
    import GPUtil

    GPUINFO = True
except:
    GPUINFO = False

from ..lib.load import inventory
from ..lib.simulation import create_simulation, store_simulation
from ..lib.timing import best_run_timer, elapsed_timer

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
    "--scenario",
    default="galaxy",
    type=str,
    show_default=True,
    help="what to simulate",
)
@click.option(
    "--scenario_param",
    default="{}",
    type=str,
    show_default=True,
    help="JSON string with scenario parameters",
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
    scenario,
    scenario_param,
    data_out_file,
    save_after_iteration,
    min_iterations,
    min_total_runtime,
    threads,
):
    """isolated single-kernel benchmark worker"""

    def _msg(**d):
        sys.stdout.write(json.dumps(d) + "\n")
        sys.stdout.flush()

    def _step():
        try:
            gc.collect()
            rt.start()
            s.step()
            rt_ = rt.stop()
            gt.start()
            gc.collect()
            gt_ = gt.stop()
        except:
            _msg(log="ERROR", msg=traceback.format_exc())
            _msg(log="EXIT", msg="BAD")
            sys.exit()
        counter[0] += 1
        if counter[0] in save_after_iteration:
            _store()
        _msg(log="STEP", runtime=rt_, gctime=gt_, counter=counter[0])
        _msg(log="BEST_TIME", value=rt.min())

    def _store():
        _msg(log="PROCEDURE", msg="Saving data after step %d ..." % counter[0])
        try:
            store_simulation(
                s,
                data_out_file,
                "kernel={kernel:s};len={n:d};step={step:d}".format(
                    kernel=kernel,
                    scenario=scenario,
                    n=len(s),
                    step=counter[0],
                ),
            )
        except:
            _msg(log="ERROR", msg=traceback.format_exc())
            _msg(log="EXIT", msg="BAD")
            sys.exit()
        _msg(log="PROCEDURE", msg="Data saved after step %d." % counter[0])

    _msg(log="START")

    counter = [0]
    scenario_param = json.loads(scenario_param)
    threads = int(threads)

    _msg(
        log="INPUT",
        simulation=dict(
            kernel=kernel,
            scenario=scenario,
            scenario_param=scenario_param,
            min_iterations=min_iterations,
            min_total_runtime=min_total_runtime,
            threads=threads,
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
            _cpu=cpuinfo.get_cpu_info() if CPUINFO else {},
            _gpu=[
                {
                    n: getattr(gpu, n)
                    for n in dir(gpu)
                    if not n.startswith("_") and n not in ("serial", "uuid")
                }
                for gpu in GPUtil.getGPUs()
            ]
            if GPUINFO
            else {},
        ),
    )

    min_total_runtime *= 10**9  # convert to ns
    inventory[kernel].load_module()

    _msg(log="PROCEDURE", msg="Creating simulation ...")
    try:
        s = create_simulation(
            scenario=scenario,
            universe_class=inventory[kernel].get_class(),
            scenario_param=scenario_param,
            threads=threads,
        )
    except:
        _msg(log="ERROR", msg=traceback.format_exc())
        _msg(log="EXIT", msg="BAD")
        sys.exit()
    _msg(log="PROCEDURE", msg="Simulation created.")
    _msg(log="SIZE", value=len(s))

    rt = best_run_timer()  # runtime
    gt = best_run_timer()  # gc time
    et = elapsed_timer()  # elapsed time

    gc.disable()

    if 0 in save_after_iteration:
        _store()

    # required min runs
    for _ in range(min_iterations):
        _step()

    # does elapsed time satisfy min_total_runtime?
    et_ = et()
    if et_ >= min_total_runtime:
        _msg(log="PROCEDURE", msg="Minimum steps sufficient.")
        _msg(log="EXIT", msg="OK")
        sys.exit()

    _msg(log="PROCEDURE", msg="Extra steps required.")
    time_remaining = min_total_runtime - et_
    iterations_remaining = time_remaining // et_ * min_iterations

    # required extra runs until min_total_runtime
    for _ in range(iterations_remaining):
        _step()

    _msg(log="EXIT", msg="OK")
    sys.exit()


def worker_command(
    data_out_file,
    interpreter,
    kernel,
    scenario,
    scenario_param,
    save_after_iteration,
    min_iterations,
    min_total_runtime,
    threads,
):
    """returns command list for use with subprocess.Popen"""
    return [
        interpreter,
        "-c",
        "from gravitation.cli import cli; cli()",
        "worker",
        "--kernel",
        "%s" % kernel,
        "--scenario",
        "%s" % scenario,
        "--scenario_param",
        "%s" % json.dumps(scenario_param),
        "--data_out_file",
        data_out_file,
        *list(
            itertools.chain(
                *[("--save_after_iteration", "%d" % it) for it in save_after_iteration]
            )
        ),
        "--min_iterations",
        "%d" % min_iterations,
        "--min_total_runtime",
        "%d" % min_total_runtime,
        "--threads",
        "%d" % threads,
    ]
