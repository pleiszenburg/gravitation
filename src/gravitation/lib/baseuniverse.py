# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/lib/baseuniverse.py: Universe base class, all kernels derive from it

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

from abc import ABC, abstractmethod
from inspect import getfile
from json import dumps, loads
from math import atan2, cos, pi, sin, sqrt
import os
from random import gauss, random, shuffle
from typing import Any, Generator, List, Optional, Tuple

from h5py import File, Group
import numpy as np

from .const import State, DEFAULT_LEN, DIMS
from .debug import typechecked
from .errors import StorageError, UniverseError
from .mass import PointMass
from .platform import Platform
from .variation import Variation

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class BaseUniverse(ABC):
    """
    kernel base class, provides infrastructure, does nothing on its own
    DERIVE FROM HERE!
    IMPLEMENT AT LEAST `step_stage1`!
    """

    _ATTRS = (
        "scale_m",
        "scale_r",
        "t",
        "T",
        "G",
        "variation",
        "platform",
        "iteration",
    )

    def __init__(
        self,
        t: float = 0.0,  # simulation start time (s)
        T: float = 1.0e3,  # simulation interval (s)
        G: float = 6.6740831e-11,  # gravitational constant
        scale_m: float = 1.0,  # scaling factor for mass (for kg)
        scale_r: float = 1.0,  # scaling factor for distances (for m)
        variation: Variation = Variation.from_default(),  # default variation
        platform: Optional[Platform] = None,  # current platform
        scaled: bool = False,
        iteration: int = 0,
        **kwargs: Any,  # catch anything else
    ):
        assert T > 0
        assert G > 0
        assert scale_m > 0
        assert scale_r > 0
        assert iteration >= 0

        self._scale_m = scale_m
        self._scale_r = scale_r
        self._t = t
        self._T = T
        self._G = G * (self._scale_r**3) / self._scale_m if not scaled else G
        self._masses = []
        self._state = State.preinit
        self._variation = variation
        self._platform = Platform.from_current() if platform is None else platform
        self._meta = kwargs
        self._iteration = int(iteration)  # HDF5 stores as int64, convert back to Python int

        self._kernel = self._get_kernel()

    def __iter__(self) -> Generator:
        yield from self._masses

    def __len__(self) -> int:
        return len(self._masses)

    def __repr__(self) -> str:
        return f"<Universe kernel={self._kernel:s} len={len(self):d} iteration={self._iteration:d} variation={str(self._variation.to_dict()):s}>"

    @property
    def variation(self) -> Variation:
        "variation of kernel"

        return self._variation

    @property
    def platform(self) -> Platform:
        "plarform"

        return self._platform

    @property
    def kernel(self) -> str:
        "kernel name"

        return self._kernel

    @property
    def state(self) -> State:
        "state of simulation"

        return self._state

    @property
    def t(self) -> float:
        "simulation time (s)"

        return self._t

    @property
    def T(self) -> float:
        "simulation interval (s)"

        return self._T

    @property
    def G(self) -> float:
        "(scaled) gravitational constant"

        return self._G

    @property
    def scale_r(self) -> float:
        "scaling factor for distances"

        return self._scale_r

    @property
    def scale_m(self) -> float:
        "scaling factor for masses"

        return self._scale_m

    @property
    def meta(self) -> dict:
        "meta data, e.g. for visualizations"

        return self._meta

    @property
    def iteration(self) -> int:
        "iteration of simulation"

        return self._iteration

    @property
    def masses(self) -> list:
        "low level access to list of masses - use carefully"

        return self._masses

    @property
    def key(self) -> str:
        "for set-like operations"

        return self.export_name_group(
            kernel=self._kernel,
            length=len(self),
            iteration=self._iteration,
            variation=self._variation.to_dict(),
            platform=self._platform.to_dict(),
        )

    def _get_kernel(self) -> str:
        "get name of kernel"

        segments = getfile(type(self)).split(os.path.sep)[::-1]
        if segments.count('kernel') != 1:
            raise UniverseError('can not safely determine name of kernel')
        return segments[segments.index('kernel') - 1]

    def add_mass(self, mass: PointMass):
        """
        add point mass object to universe
        """

        self._masses.append(mass)

    def create_mass(
        self, name: str, r: List[float], v: List[float], m: float, scaled: bool = False
    ):
        """
        create point mass object and add to universe
        """

        if self._state is State.started:
            raise UniverseError("simulation was started")
        if self._state is State.stopped:
            raise UniverseError("simulation was stopped")

        if not scaled:
            r[:] = [dim * self._scale_r for dim in r]
            v[:] = [dim * self._scale_r for dim in v]
            m *= self._scale_m

        self._masses.append(PointMass(name=name, r=r, v=v, m=m))

    def shuffle(self):
        """
        shuffle the list of mass objects
        """

        shuffle(self._masses)

    def start(self):
        """
        starts simulation
        MUST BE CALLED ONCE: AFTER ADDING OBJECTS AND BEFORE ITERATING!
        """

        if self._state is State.started:
            raise UniverseError("simulation is running")
        if self._state is State.stopped:
            raise UniverseError("simulation was stopped")

        self._state = State.started
        self.start_kernel()

    def start_kernel(self):
        """
        starts kernel, called by `start`
        REIMPLEMENT IF KERNEL-SPECIFIC INITIALIZATION IS REQUIRED!
        """

    def iterate(self, stage1: bool = True):
        """
        runs all three stages of one simulation (time-) iteration
        """

        if self._state is State.preinit:
            raise UniverseError("simulation was not started")
        if self._state is State.stopped:
            raise UniverseError("simulation was stopped")

        if stage1:
            self.push_stage1()
            self.iterate_stage1()
        self.pull_stage1()
        self.iterate_stage2()
        self.pull_stage2()
        self.iterate_stage3()

    def push_stage1(self):
        """
        REIMPLEMENT IF KERNEL-SPECIFIC DATA STRUCTURES NEED TO BE SYNCED FROM MASS OBJECTS BEFORE STAGE 1
        """

    def pull_stage1(self):
        """
        REIMPLEMENT IF KERNEL-SPECIFIC DATA STRUCTURES NEED TO BE SYNCED BACK TO MASS OBJECTS AFTER STAGE 1
        """

    def pull_stage2(self):
        """
        REIMPLEMENT IF KERNEL-SPECIFIC DATA STRUCTURES NEED TO BE SYNCED BACK TO MASS OBJECTS AFTER STAGE 2
        """

    @abstractmethod
    def iterate_stage1(self):
        """
        runs stage 1 (computes accelerations) of one simulation (time-) iteration
        MUST BE REIMPLEMENTED!
        """

        raise NotImplementedError()

    def iterate_stage2(self):
        """
        runs stage 2 (computes velocities and locations) of one simulation (time-) iteration
        """

        for pm in self._masses:
            pm.move(self._T)

    def iterate_stage3(self):
        """
        runs stage 3 (increments simulation time) of one simulation (time-) iteration, assert nan
        """

        self._t += self._T
        self._iteration += 1

    def assert_not_isnan(self):
        """
        check for NaN
        """

        for mass in self._masses:
            mass.assert_not_isnan()

    def stop(self):
        """
        stops simulation
        CAN BE CALLED ONCE: AFTER STEPPING!
        """
        if self._state is State.preinit:
            raise UniverseError("simulation was not started")
        if self._state is State.stopped:
            raise UniverseError("simulation was stopped before")

        self._state = State.stopped
        self.stop_kernel()

    def stop_kernel(self):
        """
        stops kernel, called by `stop`
        REIMPLEMENT IF KERNEL-SPECIFIC INITIALIZATION IS REQUIRED!
        """

    def gc_collect(self):
        """
        REIMPLEMENT FOR GARBAGE COLLECTION OUTSIDE OF THE PYTHON INTERPRETER'S CONTEXT
        """

    def to_hdf5(self, fn: str, gn: str):
        """
        stores simulation state into HDF5 file
        """

        with File(fn, mode = "a") as fh:
            if gn in fh.keys():
                raise StorageError("hdf5 group under this name already exists", fn, gn)
            self.to_hdf5_group(fh.create_group(gn))

    def to_hdf5_group(self, dg: Group):
        """
        stores simulation state into HDF5 group handle
        """

        if len(dg.attrs.keys()) > 0:
            raise StorageError('group attributes not empty')
        if len(dg.keys()) > 0:
            raise StorageError('group not empty')

        dtype = dict(float32 = '<f4', float64 = '<f8')[self._variation.getvalue('dtype')]

        r = dg.create_dataset("r", (len(self), DIMS), dtype=dtype)
        v = dg.create_dataset("v", (len(self), DIMS), dtype=dtype)
        m = dg.create_dataset("m", (len(self),), dtype=dtype)

        names = []
        for idx, mass in enumerate(self):
            names.append(mass.name.encode("utf-8"))
            r[idx, :] = mass.r[:]
            v[idx, :] = mass.v[:]
            m[idx] = mass.m

        buffer = np.chararray(
            (len(self),), unicode=False, itemsize=max(len(name) for name in names)
        )
        for idx, name in enumerate(names):
            buffer[idx] = name
        name = dg.create_dataset("name", (len(self),), dtype=buffer.dtype)
        name[:] = buffer[:]

        for attr in self._ATTRS:
            value = getattr(self, f"_{attr:s}")
            if attr in ('variation', 'platform'):
                value = value.to_json()
            dg.attrs[attr] = value
        for k, v in self._meta.items():
            dg.attrs[k] = v

    @classmethod
    def from_hdf5(
        cls,
        fn: str,
        gn: str,
        variation: Optional[Variation] = None,
        platform: Optional[Platform] = None,
    ):
        """
        loads simulation state from HDF5 file
        """

        with File(fn, mode = "r") as fh:
            if gn not in fh.keys():
                raise StorageError("hdf5 group under this name not present", gn)
            return cls.from_hdf5_group(fh[gn], variation = variation, platform = platform)

    @classmethod
    def from_hdf5_group(
        cls,
        dg: Group,
        variation: Optional[Variation] = None,
        platform: Optional[Platform] = None,
    ):
        """
        loads simulation state from HDF5 group handle
        """

        if len(dg.attrs.keys()) == 0:
            raise StorageError('group attributes missing')

        kwargs = {attr: dg.attrs[attr] for attr in dg.attrs.keys()}

        if variation is not None:
            kwargs['variation'] = variation
        else:
            kwargs['variation'] = Variation.from_json(kwargs['variation'])

        if platform is not None:
            kwargs['platform'] = platform
        else:
            kwargs['platform'] = Platform.from_json(kwargs['platform'])

        universe = cls(scaled=True, **kwargs)

        try:
            r, v, m, n = dg["r"], dg["v"], dg["m"], dg["name"]
        except KeyError as e:
            raise StorageError('required data missing from group') from e

        for index in range(r.shape[0]):
            universe.create_mass(
                scaled=True,
                name=bytes(n[index]).decode("utf-8"),
                r=[float(i) for i in r[index, :]],
                v=[float(i) for i in v[index, :]],
                m=float(m[index]),
            )

        return universe

    @staticmethod
    def export_name_group(**kwargs) -> str:
        "generate name for HDF5 group"

        return dumps(kwargs, sort_keys=True)

    @staticmethod
    def import_name_group(raw: str) -> dict:
        "parse name of HDF5 group"

        return loads(raw)

    @classmethod
    def from_galaxy(
        cls,
        T: float = 2.0e12,
        scale_m: float = 1.0e-30,
        scale_r: float = 1.0e-10,
        variation: Variation = Variation.from_default(),
        platform: Optional[Platform] = None,
        length: int = DEFAULT_LEN,
        r: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        v: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        g_alpha: float = 0.0,
        g_beta: float = 0.0,
        m_hole: float = 4e40,
        m_star: float = 2e30,
        radius: float = 1e20,
    ):
        """
        creates a galaxy-like bunch of objects and adds them to universe
        """

        universe = cls(
            T=T,
            scale_m=scale_m,
            scale_r=scale_r,
            variation=variation,
            platform=Platform.from_current() if platform is None else platform,
        )

        universe.create_mass(
            name="back hole",
            r=[d for d in r],
            v=[d for d in v],
            m=m_hole,
        )
        universe.masses[-1].assert_not_isnan()

        for n in range(length - 1):
            alpha = random() * 2.0 * pi

            # generate disk of stars
            if n < (length * 4 // 5):
                # random orbit radius
                r_abs = (random() * 4.5 + 0.1) * radius

                # compute star position
                r_s = [
                    r_abs * cos(alpha),
                    r_abs * sin(alpha),
                    (0.5 * random() - 0.25)
                    * radius
                    * ((4.5 + 0.1) * radius - r_abs)
                    / ((4.5 + 0.1) * radius),
                ]

                name = "disk star"

            # generate central cloud of stars
            else:
                # random orbit radius
                r_abs = (random() * 0.75 + 0.1) * radius

                # random inclination
                beta = pi * (random() - 0.5)

                # compute star position
                r_s = [
                    r_abs * cos(alpha) * cos(beta),
                    r_abs * sin(alpha) * cos(beta),
                    r_abs * sin(beta),
                ]

                name = "cloud star"

            # absolute orbital velocity around central body
            v_abs = sqrt(universe.G * m_hole / sqrt(sum([d**2 for d in r_s])))
            # phase angle
            v_alpha = alpha - (pi / 2)

            # preliminary velocity vector
            v_s = [v_abs * cos(v_alpha), v_abs * sin(v_alpha), 0.0]

            # rotate around x axis (beta)
            v_s[1:] = [v_s[1] * cos(g_beta), v_s[1] * sin(g_beta)]

            # rotate around z axis (alpha)
            v_alpha = atan2(v_s[1], v_s[0]) + g_alpha
            v_factor = sqrt(v_s[0] ** 2 + v_s[1] ** 2)
            v_s[0:2] = [v_factor * cos(v_alpha), v_factor * sin(v_alpha)]

            # actual velocity vector
            v_s = [d + e for d, e in zip(v_s, v)]

            # rotate around x axis (beta)
            r_beta = atan2(r_s[2], r_s[1]) + g_beta
            r_factor = sqrt(r_s[2] ** 2 + r_s[1] ** 2)
            r_s[1:] = [r_factor * cos(r_beta), r_factor * sin(r_beta)]

            # rotate around z axis (alpha)
            r_alpha = atan2(r_s[1], r_s[0]) + g_alpha
            r_factor = sqrt(r_s[0] ** 2 + r_s[1] ** 2)
            r_s[0:2] = [r_factor * cos(r_alpha), r_factor * sin(r_alpha)]

            # shift by center of galaxy
            r_s = [d + e for d, e in zip(r_s, r)]

            m_star_rand = m_star * 10 ** gauss(0.0, 1.0)

            universe.create_mass(
                name=name, r=r_s, v=v_s, m=m_star_rand,
            )
            universe.masses[-1].assert_not_isnan()

        universe.shuffle()

        return universe
