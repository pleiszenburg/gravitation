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

import sys

import click
import plotly.graph_objs as go

from ._kernel import add_kernel_commands, add_platform_options
from ..lib.errors import VariationError, VerificationError
from ..lib.kernel import Kernel
from ..lib.platform import Platform
from ..lib.verification import Verification

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONST
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@click.option(
    "--iteration",
    "-i",
    default=10,
    type=int,
    show_default=True,
    required=True,
    help="iteration for verification",
)
@click.argument(
    "datafile",
    type=click.File("rb"),
    nargs=1,
)
@click.argument(
    "plot",
    type=click.File("w"),
    nargs=1,
)
@click.pass_context
def verify(
    ctx,
    iteration,
    datafile,
    plot,
    **platform_kwargs,
):
    """verify kernel results"""

    def run(kernel: Kernel, **variation_kwargs):

        kernel.load_meta()
        try:
            kernel.variations.select(**variation_kwargs)
        except VariationError as e:
            kernel.variations.print()
            print('ERROR:', e)
            sys.exit(1)

        try:
            with Verification(datafile) as verification:
                figure = verification.to_verify_figure(
                    kernel = kernel.name,
                    variation = kernel.variations.selected,
                    platform = Platform.from_dict(**platform_kwargs),
                    iteration = iteration,
                )
        except VerificationError:
            sys.exit(1)

        fig = go.Figure(figure)
        fig.write_html(plot)
        fig.show()

    ctx.meta['run'] = run  # runs via kernel sub-command


add_platform_options(verify)
verify = click.group(short_help="plot results of model verification against reference kernel")(verify)
add_kernel_commands(verify)
