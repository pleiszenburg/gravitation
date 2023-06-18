# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/lib/verification.py: Verify simulation results

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

from math import log2, sqrt
from typing import Any, List, Optional

from h5py import File, Group

from .baseuniverse import BaseUniverse
from .debug import typechecked
from .errors import VerificationError
from .platform import Platform
from .variation import Variation
from .zerouniverse import UniverseZero

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class Verification:
    "verify simulation results"

    def __init__(self, fn: str):
        self._fn = fn
        self._f = None
        self._snapshots = None

    def __enter__(self):
        if self.is_open:
            raise VerificationError('data file already open')
        self._f = File(self._fn, mode = 'r')
        self._snapshots = [
            BaseUniverse.import_name_group(key)
            for key in self._f.keys()
        ]
        return self

    def __exit__(self, *args: Any, **kwargs: Any):
        if not self.is_open:
            raise VerificationError('data file already closed')
        self._f.close()
        self._f = None
        self._snapshots = None

    @property
    def is_open(self) -> bool:
        "is data file open?"

        return self._f is not None

    def _get_group(
        self,
        kernel: str,
        length: int,
        iteration: int,
        variation: Variation,
        platform: Platform,
    ) -> Optional[Group]:
        "find hdf5 group in file"

        target = dict(
            kernel = kernel,
            length = length,
            iteration = iteration,
            variation = variation.to_dict(),
            platform = platform.to_dict(),
        )

        if target not in self._snapshots:
            return None

        return self._f[BaseUniverse.export_name_group(**target)]

    def _verify_pair(
        self,
        reference: Group,
        target: Group,
    ) -> List[float]:

        reference_universe = UniverseZero.from_hdf5_group(reference)
        target_universe = UniverseZero.from_hdf5_group(target)

        return [
            sqrt(sum([
                (ar - br) ** 2
                for ar, br in zip(a.r, b.r)
            ]))
            for a, b in zip(reference_universe, target_universe)
        ]

    def verify(
        self,
        kernel: str,
        iteration: int,
        variation: Variation,
        platform: Platform,
    ) -> list:
        "verify data against reference"

        if not self.is_open:
            raise VerificationError('data file not open')

        lengths = sorted({
            snapshot['length']
            for snapshot in self._snapshots
            if snapshot['kernel'] != 'zero'
        })
        target_kernels = sorted({
            snapshot['kernel']
            for snapshot in self._snapshots
            if snapshot['kernel'] != 'zero'
        })
        target_platforms = {
            Platform.from_dict(**snapshot['platform'])
            for snapshot in self._snapshots
            if snapshot['kernel'] != 'zero'
        }

        results = []

        for target_kernel in target_kernels:

            target_variations = {
                Variation.from_dict(**snapshot['variation'])
                for snapshot in self._snapshots
                if snapshot['kernel'] == target_kernel
            }

            for target_variation in target_variations:

                for target_platform in target_platforms:

                    if all((
                        kernel == target_kernel,
                        variation == target_variation,
                        platform == target_platform,
                    )):
                        continue

                    dists = []
                    length_labels = []
                    name = f'{target_kernel:s} {repr(target_variation):s}'  # TODO platform, sep by "/" as label? Label API?

                    for length in lengths:

                        reference_group = self._get_group(
                            kernel = kernel,
                            length = length,
                            iteration = iteration,
                            variation = variation,
                            platform = platform,
                        )
                        target_group = self._get_group(
                            kernel = target_kernel,
                            length = length,
                            iteration = iteration,
                            variation = target_variation,
                            platform = target_platform,
                        )

                        if reference_group is None or target_group is None:
                            print('data missing')
                            continue

                        dist = self._verify_pair(
                            reference_group,
                            target_group,
                        )

                        print(
                            f"Matching {name:s}: length={length:d} iteration={iteration:d} min={min(dist):0.02e} max={max(dist):0.02e}"
                        )

                        length_label = f"2^{round(log2(length)):d}"
                        length_labels.extend(length_label for _ in dist)
                        dists.extend(dist)

                    results.append(dict(
                        lengths = length_labels,
                        dists = dists,
                        name = name,
                    ))

        return results

    def to_verify_figure(self, *args, iteration: int = 0, **kwargs) -> dict:
        "verify data against reference and provide dict for plotting"

        return dict(
            data = [
                dict(
                    type='box',
                    x=data['lengths'],
                    y=data['dists'],
                    name=data['name'],
                )
                for data in self.verify(*args, iteration = iteration, **kwargs)
            ],
            layout = dict(
                xaxis_title="items per simulation",
                yaxis_title=f"location offset iteration={iteration:d}",
                yaxis_type="log",
                boxmode="group",
                showlegend = True,
            ),
        )
