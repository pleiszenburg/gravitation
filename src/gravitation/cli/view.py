# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/view.py: view command

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
from ..lib.errors import VariationError
from ..lib.kernel import Kernel
from ..lib.view import VIEWS

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@click.group(short_help="view a simulation progressing in realtime")
@click.option(
    "--len",
    default=DEFAULT_LEN,
    type=int,
    show_default=True,
    help="number of point masses",
)
@click.option(
    "--iterations_per_frame",
    default=1,
    type=int,
    show_default=True,
    help="simulation steps per frame",
)
@click.option(
    "--max_iterations",
    default=-1,
    type=int,
    show_default=True,
    help="maximum number of simulation steps, no maximum if -1",
)
@click.option(
    "--timer_buffer",
    default=20,
    type=int,
    show_default=True,
    help="average time measurements over this many samples",
)
@click.option(
    "--backend",
    default=sorted(VIEWS.keys())[-1],
    type=click.Choice(sorted(VIEWS.keys())),
    show_default=True,
    help="view backend",
)
@click.pass_context
def view(
    ctx,
    len: int,
    iterations_per_frame: int,
    max_iterations: int,
    timer_buffer: int,
    backend: str,
):
    """view a simulation progressing in realtime"""

    def run(kernel: Kernel, **kwargs):

        kernel.load_meta()
        try:
            kernel.variations.select(**kwargs)
        except VariationError as e:
            kernel.variations.print()
            print('ERROR:', e)
            sys.exit(1)

        VIEWS[backend].load_cls()
        viewer = VIEWS[backend].cls(
            kernel = kernel.name,
            length = len,
            variation = kernel.variations.selected,
            timer_buffer = timer_buffer,
            iterations_per_frame = iterations_per_frame,
            max_iterations = None if max_iterations == -1 else max_iterations,
        )
        viewer.run()

    ctx.meta['run'] = run  # runs via kernel sub-command


add_kernel_commands(view)
