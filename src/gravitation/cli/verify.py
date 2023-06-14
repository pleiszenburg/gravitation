# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/verify.py: verify command

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

import atexit
from math import log2

import click
import h5py
import numpy as np
from plotly.offline import plot as _plot
import plotly.graph_objs as go

from ..lib.baseuniverse import BaseUniverse
from ..lib.kernel import KERNELS

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONST
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@click.command(short_help="plot results of model verification against reference kernel")
@click.argument(
    "reference",
    type=click.Choice(sorted(list(KERNELS.keys()))),
    required=True,
)
@click.option(
    "--datafile",
    "-d",
    default="data.h5",
    type=str,
    show_default=True,
    help="name of benchmark data output file",
)
@click.option(
    "--out",
    "-o",
    default="verify.html",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    show_default=True,
    required=True,
    help="name of output html file",
)
def verify(
    reference: str,
    datafile: str,
    out: str,
):
    """verify kernel results"""

    f = h5py.File(datafile, mode="r")
    atexit.register(f.close)

    runs = (BaseUniverse.import_name_group(key) for key in f.keys())
    runs = [run for run in runs if run['kernel'] != 'zero']

    kernels = sorted({meta["kernel"] for meta in runs})
    lens = sorted({meta["len"] for meta in runs})
    step = max({meta["step"] for meta in runs})

    if reference not in kernels:
        raise ValueError("no data present for reference kernel", reference)

    x = []

    for len_ in lens:
        key = f"2^{round(log2(len_)):d}"
        x.extend([key for _ in range(len_)])

    data = {}

    for kernel in kernels:
        if kernel == reference:
            continue

        data[kernel] = []

        for len_ in lens:
            reference_key = BaseUniverse.export_name_group(
                kernel=reference, len=len_, step=step
            )
            kernel_key = BaseUniverse.export_name_group(
                kernel=kernel, len=len_, step=step
            )

            if reference_key not in f.keys():
                print(f"No data for reference kernel {reference_key:s}")
            if kernel_key not in f.keys():
                print(f"No data for target target {kernel_key:s}")
            if reference_key not in f.keys() or kernel_key not in f.keys():
                data[kernel].extend(None for _ in range(len_))
                continue

            reference_r = f[reference_key]["r"][...]
            kernel_r = f[kernel_key]["r"][...]
            dist = np.sqrt(np.add.reduce((kernel_r - reference_r) ** 2, axis=1))
            assert dist.shape == (len_,)

            print(
                f"Matching {kernel:s}: len={len_:d} step={step:d} min={dist.min():0.02e} max={dist.max():0.02e} mean={dist.mean():0.02e}"
            )

            data[kernel].extend([float(n) for n in dist])

    fig = go.Figure()

    for kernel in kernels:
        if kernel == reference:
            continue

        assert len(x) == len(data[kernel])

        fig.add_trace(
            go.Box(
                x=x,
                y=data[kernel],
                name=kernel,
            )
        )

    fig.update_layout(
        xaxis_title="items per simulation",
        yaxis_title=f"location offset step={step:d}",
        yaxis_type="log",
        boxmode="group",
    )

    _plot(fig, filename=out)
