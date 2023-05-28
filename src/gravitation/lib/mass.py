# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/lib/mass.py: Point mass class

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
# IMPORTS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from math import isnan
from typing import List

from .const import DIMS
from .debug import typechecked

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@typechecked
class PointMass:
    """
    holds point mass description
    """

    def __init__(self, name: str, r: List[float], v: List[float], m: float):
        assert len(r) == DIMS
        assert len(v) == DIMS

        self._name, self._r, self._v, self._a, self._m = (
            name,
            r,
            v,
            [0.0 for _ in range(len(r))],
            m,
        )

    def __repr__(self):
        return (
            "<PointMass name={name} | "
            "{x:.4e}, {y:.4e}, {z:.4e} | "
            "{vx:.4e}, {vy:.4e}, {vz:.4e}>"
        ).format(
            name=self._name,
            **{key: val for key, val in zip(["x", "y", "z"], self._r)},
            **{key: val for key, val in zip(["vx", "vy", "vz"], self._v)},
        )

    @property
    def name(self) -> str:
        "name"

        return self._name

    @property
    def r(self) -> List[float]:
        "location"

        return self._r

    @property
    def v(self) -> List[float]:
        "velocity"

        return self._v

    @property
    def a(self) -> List[float]:
        "acceleration"

        return self._a

    @property
    def m(self) -> float:
        "mass"

        return self._m

    def assert_not_isnan(self):
        "check for NaN"

        assert not any(isnan(d) for d in self._r)
        assert not any(isnan(d) for d in self._v)
        assert not any(isnan(d) for d in self._a)
        assert not isnan(self._m)

    def move(self, T: float):
        """
        moves with precomputed acceleration (base stage 2 implementation)
        """

        self._v[:] = [a * T + v for v, a in zip(self._v, self._a)]
        self._r[:] = [v * T + r for r, v in zip(self._r, self._v)]
        self._a[:] = [0.0 for _ in range(len(self._a))]