# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/_base_.py: Base class, all kernels derive from it

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
# CONST
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

STATE_PREINIT = 0
STATE_STARTED = 1
STATE_STOPPED = 2

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class _point_mass:
    """holds point mass description"""

    def __init__(self, name, r, v, m):
        self._name, self._r, self._v, self._a, self._m = (
            name,
            r,
            v,
            [0.0 for _ in range(len(r))],
            m,
        )

    def __str__(self):
        return (
            "{name} | " "{x:.4e}, {y:.4e}, {z:.4e} | " "{vx:.4e}, {vy:.4e}, {vz:.4e}"
        ).format(
            name=self._name,
            **{key: val for key, val in zip(["x", "y", "z"], self._r)},
            **{key: val for key, val in zip(["vx", "vy", "vz"], self._v)},
        )

    def move(self, T):
        """moves with precomputed acceleration (base stage 2 implementation)"""
        self._v[:] = [a * T + v for v, a in zip(self._v, self._a)]
        self._r[:] = [v * T + r for r, v in zip(self._r, self._v)]
        self._a[:] = [0.0 for _ in range(len(self._a))]


class universe_base:
    """kernel base class, provides infrastructure, does nothing on its own
    DERIVE FROM HERE!
    IMPLEMENT / OVERLOAD AT LEAST `step_stage1`!"""

    def __init__(
        self,
        t=0.0,  # simulation start time (s)
        T=1.0e3,  # simulation interval (s)
        G=6.6740831e-11,  # gravitational constant
        scale_m=1.0,  # scaling factor for mass (for kg)
        scale_r=1.0,  # scaling factor for distances (for m)
        dtype="float32",  # datatype for numerical computations
        threads=1,  # maximum number of threads
        **kwargs,  # catch anything else
    ):
        """MUST NOT BE OVERLOADED!"""
        self._scale_m = scale_m
        self._scale_r = scale_r
        self._t = t
        self._T = T
        if not kwargs.pop("scale_off", False):
            self._G = G * (self._scale_r**3) / self._scale_m
        else:
            self._G = G
        self._mass_list = []
        self._state = STATE_PREINIT
        self._dtype = dtype
        self._threads = threads
        self._meta = kwargs

    def __iter__(self):
        """MUST NOT BE OVERLOADED!"""
        return (p for p in self._mass_list)

    def __len__(self):
        """MUST NOT BE OVERLOADED!"""
        return len(self._mass_list)

    def __str__(self):
        """CAN BE OVERLOADED!"""
        return "\n".join([str(pm) for pm in self._mass_list])

    def add_object(self, **kwargs):
        """adds point mass object to kernel object
        MUST NOT BE OVERLOADED!"""
        if self._state == STATE_STARTED:
            raise SyntaxError("simulation was started")
        if self._state == STATE_STOPPED:
            raise SyntaxError("simulation was stopped")
        if not kwargs.pop("scale_off", False):
            kwargs["r"][:] = [dim * self._scale_r for dim in kwargs["r"]]
            kwargs["v"][:] = [dim * self._scale_r for dim in kwargs["v"]]
            kwargs["m"] *= self._scale_m
        self._mass_list.append(_point_mass(**kwargs))

    def start(self):
        """starts simulation
        MUST BE CALLED ONCE: AFTER ADDING OBJECTS AND BEFORE STEPPING!
        MUST NOT BE OVERLOADED!"""
        if self._state == STATE_STARTED:
            raise SyntaxError("simulation is running")
        if self._state == STATE_STOPPED:
            raise SyntaxError("simulation was stopped")
        self._state = STATE_STARTED
        self.start_kernel()

    def start_kernel(self):
        """starts kernel, called by "start"
        OVERLOAD IF KERNEL-SPECIFIC INITIALIZATION IS REQUIRED!"""
        pass

    def step(self):
        """runs all three stages of one simulation (time-) step
        MUST NOT BE OVERLOADED!"""
        if self._state == STATE_PREINIT:
            raise SyntaxError("simulation was not started")
        if self._state == STATE_STOPPED:
            raise SyntaxError("simulation was stopped")
        self.step_stage1()
        self.step_stage2()
        self.step_stage3()

    def step_stage1(self):
        """runs stage 1 (computes accelerations) of one simulation (time-) step
        MUST BE OVERLOADED!"""
        raise NotImplementedError()

    def step_stage2(self):
        """runs stage 2 (computes velocities and locations) of one simulation (time-) step
        CAN BE OVERLOADED!"""
        for pm in self._mass_list:
            pm.move(self._T)

    def step_stage3(self):
        """runs stage 3 (increments simulation time) of one simulation (time-) step
        MUST NOT BE OVERLOADED!"""
        self._t += self._T

    def stop(self):
        """stops simulation
        CAN BE CALLED ONCE: AFTER STEPPING!
        MUST NOT BE OVERLOADED!"""
        if self._state == STATE_PREINIT:
            raise SyntaxError("simulation was not started")
        if self._state == STATE_STOPPED:
            raise SyntaxError("simulation was stopped before")
        self._state = STATE_STOPPED
        self.stop_kernel()

    def stop_kernel(self):
        """stops kernel, called by "stop"
        OVERLOAD IF KERNEL-SPECIFIC DESTRUCTION IS REQUIRED!"""
        pass
