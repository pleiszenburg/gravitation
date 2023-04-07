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

import click
import h5py
import numpy as np

from ..kernel._base import UniverseBase
from ..lib.load import inventory

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONST
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@click.command(short_help="verify model results")
@click.option(
    "--kernel",
    "-k",
    type=click.Choice(sorted(list(inventory.keys()))),
    required=True,
    help="name of base kernel module",
)
@click.option(
    "--data_out_file",
    "-o",
    default="data_out.h5",
    type=str,
    show_default=True,
    help="name of inout data file",
)
def verify(
    kernel: str,
    data_out_file: str,
):
    """verify kernel results"""

    f = h5py.File(data_out_file, mode = 'r')
    atexit.register(f.close)

    runs = [UniverseBase.import_name_group(key) for key in f.keys()]

    kernels = sorted({meta['kernel'] for meta in runs})
    lens = sorted({meta['len'] for meta in runs})
    steps = sorted({meta['step'] for meta in runs})

    if kernel not in kernels:
        raise ValueError('no data present for base kernel', kernel)

    for target in kernels:
        if target == kernel:
            continue
        for len_ in lens:
            for step in steps:

                source_key = UniverseBase.export_name_group(kernel = kernel, len = len_, step = step)
                if source_key not in f.keys():
                    print(f'No data for source {source_key:s}')
                    continue

                target_key = UniverseBase.export_name_group(kernel = target, len = len_, step = step)
                if target_key not in f.keys():
                    print(f'No data for target {target_key:s}')
                    continue

                print(f'Match {kernel:s} against {target:s}: len={len_:d} step={step:d}')

                source_r = f[source_key]['r'][...]
                target_r = f[target_key]['r'][...]
                dist = np.sqrt(np.add.reduce((target_r - source_r) ** 2, axis = 1))

                print(dist.min(), dist.max(), dist.mean())
