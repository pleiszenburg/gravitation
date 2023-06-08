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
from typing import List

import numpy as np
from plotly.offline import plot as _plot
import plotly.graph_objs as go

import click

from ..lib.debug import typechecked

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class _Benchmark:
    "handle data from analyzed log"

    def __init__(self, meta: dict, runtime: List[int], gctime: List[int]):
        self._meta = meta
        self._runtime = runtime
        self._gctime = gctime

    def __len__(self) -> int:
        return self._meta['simulation']['length']

    def get_key(self, length: bool = True) -> str:
        "unique key for benchmark kernel variation minus length"

        variation = [
            f"{key:s}={self._meta['kernel'][key]:s}"
            for key in sorted(self._meta['kernel'].keys())
        ]
        label = [
            f"kernel={self._meta['kernel']['name']:s}",
            *variation,
            f"implementation={self._meta['python']['implementation']:s}",
        ]
        if length:
            label.append(f"len={self._meta['simulation']['length']:d}")

        return " / ".join(label)

    @property
    def best_runtime(self) -> int:
        "best runtime in ns"

        return min(self._runtime)

    @property
    def best_gctime(self) -> int:
        "best gc time in ns"

        return min(self._gctime)


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
    "--out",
    "-o",
    default="benchmark.html",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    show_default=True,
    required=True,
    help="name of output html file",
)
def plot(logfile, out):
    """plot benchmark json data file"""

    logs = []
    for fh in logfile:
        logs.extend(json.loads(fh.read()))

    benchmarks = [_Benchmark(**log) for log in logs]

    runtimes = {
        key: dict() for key in {
            benchmark.get_key(length = False) for benchmark in benchmarks
        }
    }
    for benchmark in benchmarks:
        runtimes[benchmark.get_key(length = False)][len(benchmark)] = benchmark.best_runtime

    traces = []
    xc = set()
    yc = set()
    for key, results in sorted(runtimes.items(), key=lambda x: x[0]):
        x, y = [], []
        for size, runtime in sorted(results.items(), key=lambda x: x[0]):
            x.append(size)
            y.append(runtime * 1e-9)
        traces.append(go.Scatter(
            x=x,
            y=y,
            name=key,
            mode="lines+markers",
            hovertemplate = "%{y}",
        ))
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

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    fig.update_layout(hovermode="x unified")

    _plot(fig, filename=out)
