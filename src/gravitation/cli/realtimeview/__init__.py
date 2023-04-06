# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/realtimeview/__init__.py: realtimeview command

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

import importlib
import json
import os

import click
import psutil

from ...lib.load import inventory

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONST
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

MAX_TREADS = psutil.cpu_count(logical=True)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def _get_backends():
    """auto-detects backends"""
    return sorted(
        [
            item[8:-3] if item.lower().endswith(".py") else item[8:]
            for item in os.listdir(os.path.dirname(__file__))
            if item.startswith("backend_")
        ]
    )


@click.command(short_help="view a simulation progressing in realtime")
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
    "--steps_per_frame",
    default=-1,
    type=int,
    show_default=True,
    help="simulation steps per frame, use scenario default if -1",
)
@click.option(
    "--max_iterations",
    default=-1,
    type=int,
    show_default=True,
    help="maximum number of simulation steps, no maximum if -1",
)
@click.option(
    "--backend",
    default=_get_backends()[0],
    type=click.Choice(_get_backends()),
    show_default=True,
    help="plot backend",
)
@click.option(
    "--threads",
    "-p",
    default="1",
    type=click.Choice([str(i) for i in range(1, MAX_TREADS + 1)]),
    show_default=True,
    help="number of threads/processes for parallel implementations",
)
def realtimeview(
    kernel, scenario, scenario_param, steps_per_frame, max_iterations, backend, threads
):
    """view a simulation progressing in realtime"""

    scenario_param = json.loads(scenario_param)
    threads = int(threads)

    view_param_pos = (
        kernel,
        threads,
        scenario,
        scenario_param,
    )
    view_param_kw = {
        "steps_per_frame": None if steps_per_frame == -1 else steps_per_frame,
        "max_iterations": None if max_iterations == -1 else max_iterations,
    }

    viewer = importlib.import_module(
        "gravitation.cli.realtimeview.backend_%s" % backend
    ).realtimeview

    viewer(*view_param_pos, **view_param_kw).loop()
