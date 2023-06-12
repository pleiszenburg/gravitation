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

import sys

import click

from ._kernel import add_kernel_commands
from ..lib.const import DEFAULT_LEN
from ..lib.errors import VariationError, WorkerError
from ..lib.kernel import Kernel
from ..lib.worker import Worker

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

        try:
            Worker(
                kernel = kernel.name,
                variation = kernel.variations.selected,
                length = len,
                datafile = datafile,
                save_after_iteration = save_after_iteration,
                read_initial_state = read_initial_state,
                min_iterations = min_iterations,
                min_total_runtime = min_total_runtime,
            ).run()
        except WorkerError:
            sys.exit(1)

    ctx.meta['run'] = run  # runs via kernel sub-command


add_kernel_commands(worker)
