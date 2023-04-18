# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/plot.py: plot command

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

import json

import numpy as np
from plotly.offline import plot as _plot
import plotly.graph_objs as go

import click

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@click.command(short_help="plot benchmark json data file")
@click.option(
    "--logfile",
    "-l",
    default=["benchmark.json"],
    type=click.File("r"),
    show_default=True,
    multiple=True,
    required=True,
    help="name of input log file, can be specified multiple times",
)
@click.option(
    "--html_out",
    "-o",
    default="benchmark.html",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    show_default=True,
    required=True,
    help="name of output html file",
)
def plot(logfile, html_out):
    """plot benchmark json data file"""

    logs = []
    for fh in logfile:
        logs.extend(json.loads(fh.read()))

    runtimes = {
        item: dict() for item in {item["meta"]["simulation"]["kernel"] for item in logs}
    }

    for item in logs:
        runtimes[item["meta"]["simulation"]["kernel"]][
            item["meta"]["simulation"]["size"]
        ] = min(item["runtime"])

    traces = []
    xc = set()
    yc = set()
    for kernel_name, kernel_results in sorted(runtimes.items(), key=lambda x: x[0]):
        x, y = [], []
        for size, runtime in sorted(kernel_results.items(), key=lambda x: x[0]):
            x.append(size)
            y.append(runtime / 1e9)
        traces.append(go.Scatter(x=x, y=y, name=kernel_name, mode="lines+markers"))
        xc.update(set(x))
        yc.update(set(y))

    x = np.array(sorted(xc), dtype = 'f8')
    x0 = x[0]
    y0 = min(yc)
    for idx in range(2, 4):
        traces.append(go.Scatter(
            x=x,
            y=(y0 / (x0 ** idx)) * (x ** idx),
            name=f'x^{idx:d}',
            mode="lines",
            line_color='rgba(200,200,200,128)',
        ))

    layout = go.Layout(
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
    )
    fig = go.Figure(data=traces, layout=layout)
    _plot(fig, filename=html_out)
