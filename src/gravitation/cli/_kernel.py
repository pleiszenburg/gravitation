# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/_kernel.py: kernel command line entry points

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

import click

from ..lib.kernel import KERNELS
from ..lib.platform import Platform

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def add_kernel_commands(command):
    """
    auto-detects kernels and turns them into sub-commands

    will run: `ctx.meta['run'](kernel, **kwargs)`
    """

    for name in sorted(KERNELS.keys()):
        command.add_command(_make_kernel_command(name))


def _make_kernel_command(name: str):
    """
    dynamically generates thin wrapper command around kernel by name, does not import kernel
    """

    KERNELS[name].load_meta()

    def command(ctx, **kwargs):
        ctx.meta['run'](KERNELS[name], **kwargs)

    command.__name__ = name
    command.__doc__ = KERNELS[name].description

    command = click.pass_context(command)

    for option in KERNELS[name].variations.to_options():
        command = click.option(
            f"--{option.name:s}",
            type = click.Choice([str(choice) for choice in option.choices]),
            required = True,
            default = str(option.choices[0])
        )(command)

    command = click.command(
        short_help = KERNELS[name].description
    )(command)

    return command


def add_platform_options(command):
    """
    add options based on current platform
    """

    options = Platform.from_current().to_options()

    for option in options:
        command = click.option(
            f"--{option.name:s}",
            type = option.type,
            default = str(option.choices[0]),
            show_default = True,
            required = True,
        )(command)
