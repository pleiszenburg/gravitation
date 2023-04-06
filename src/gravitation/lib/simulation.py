# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/lib/simulation.py: Simulation infrastructure (load, store, create)

	Copyright (C) 2019 Sebastian M. Ernst <ernst@pleiszenburg.de>

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

import math
import random

import numpy as np
import h5py

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def create_simulation(scenario, universe_class, scenario_param=None, threads=1):
    """creates simulation based in scenario name and kernel class"""

    universe_param = {"threads": threads}
    scenario_param = scenario_param if scenario_param is not None else {}

    if scenario == "solarsystem":
        universe_param.update(scenario_param)
        universe_obj = universe_class(**universe_param)
        universe_obj._screen = {
            "unit": 149597870700,  # m == 1 AU
            "unit_size": [3.0, 2.25],  # units
            "average_over_steps": 500,
            "steps_per_frame": 20,
        }
        create_solarsystem(universe_obj)
    elif scenario == "galaxy":
        universe_param.update(
            {
                "T": 2.0e12,
                "scale_m": 1.0e-30,
                "scale_r": 1.0e-10,
                "dtype": "float32",
            }
        )
        universe_param.update(scenario_param)
        universe_obj = universe_class(**universe_param)
        universe_obj._screen = {
            "unit": 1e20,  # m
            "unit_size": [16.0, 10.0],  # units
            "average_over_steps": 20,
            "steps_per_frame": 1,
        }
        create_galaxy(
            universe_obj=universe_obj,
            stars_len=scenario_param.get("stars_len", 2000) - 1,
            r=[0.0, 0.0, 0.0],
            v=[0.0, 0.0, 0.0],
            g_alpha=0.0,
            g_beta=0.0,
            m_hole=4e40,
            m_star=2e30,
            radius=1e20,
        )
    else:
        raise ValueError('Unknown scenario: "%s"' % scenario)

    universe_obj.start()

    return universe_obj


def create_solarsystem(universe_obj):
    """adds (some) solar system objects to simulation / kernel object"""

    universe_obj.add_object(
        name="sun", r=[0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0], m=1.98892e30
    )
    universe_obj.add_object(
        name="earth",
        r=[0.0, -149597870700.0, 0.0],
        v=[29777.777, 0.0, 0.0],
        m=5.97237e24,
    )


def create_galaxy(
    universe_obj, stars_len, r, v, g_alpha, g_beta, m_hole, m_star, radius
):
    """adds a galaxy-like bunch of objects to simulation / kernel object"""

    universe_obj.add_object(
        name="back hole", r=[d for d in r], v=[d for d in v], m=m_hole
    )

    for n in range(stars_len):
        alpha = random.random() * 2.0 * math.pi

        if n < (stars_len * 4 // 5):
            # Zufälliger Bahnradius
            r_abs = (random.random() * 4.5 + 0.1) * radius

            # Position des Sterns berechnen
            r_s = [
                r_abs * math.cos(alpha),
                r_abs * math.sin(alpha),
                (0.5 * random.random() - 0.25)
                * radius
                * ((4.5 + 0.1) * radius - r_abs)
                / ((4.5 + 0.1) * radius),
            ]

        # Den zentralen Sternenhaufen der Galaxie generieren
        else:
            # Zufälliger Bahnradius
            r_abs = (random.random() * 0.75 + 0.1) * radius

            # Zufällige Inklination (wird nur für Position verwendet)
            beta = math.pi * (random.random() - 0.5)

            # Position des Sterns berechnen
            r_s = [
                r_abs * math.cos(alpha) * math.cos(beta),
                r_abs * math.sin(alpha) * math.cos(beta),
                r_abs * math.sin(beta),
            ]

        # Hilfsgrößen für Geschwindigkeitsvektor generieren für Kreisbahn um schwarzes Loch
        v_abs = math.sqrt(
            universe_obj._G * m_hole / math.sqrt(sum([d**2 for d in r_s]))
        )
        v_alpha = alpha - (
            math.pi / 2
        )  # Phasenverschiebung [Drehung] um -90° (von ausgehend 0 bis 360°): Geschwindigkeitsvektor im rechten Winkel zur Bahn

        # Vorläufiger Geschwindigkeitsvektor (Galaxie passend drehen: Geschwindigkeit)
        v_s = [v_abs * math.cos(v_alpha), v_abs * math.sin(v_alpha), 0.0]

        # Um X-Achse drehen (beta)
        v_s[1:] = [v_s[1] * math.cos(g_beta), v_s[1] * math.sin(g_beta)]

        # Um Z-Achse drehen (alpha)
        v_alpha = math.atan2(v_s[1], v_s[0]) + g_alpha
        v_factor = math.sqrt(v_s[0] ** 2 + v_s[1] ** 2)
        v_s[0:2] = [v_factor * math.cos(v_alpha), v_factor * math.sin(v_alpha)]

        # Geschwindigkeitsvektor setzen
        v_s = [d + e for d, e in zip(v_s, v)]

        # Rotate galaxy: x axis (beta)
        r_beta = math.atan2(r_s[2], r_s[1]) + g_beta
        r_factor = math.sqrt(r_s[2] ** 2 + r_s[1] ** 2)
        r_s[1:] = [r_factor * math.cos(r_beta), r_factor * math.sin(r_beta)]

        # Rotate galaxy: z axis (alpha)
        r_alpha = math.atan2(r_s[1], r_s[0]) + g_alpha
        r_factor = math.sqrt(r_s[0] ** 2 + r_s[1] ** 2)
        r_s[0:2] = [r_factor * math.cos(r_alpha), r_factor * math.sin(r_alpha)]

        # Galaxie gemäß den Koordninaten verschieben
        r_s = [d + e for d, e in zip(r_s, r)]

        universe_obj.add_object(name="star", r=r_s, v=v_s, m=m_star)


def load_simulation(universe_class, fn, gn, threads=None):
    """loads simulation from HDF5 file into object generated from kernel class"""

    f = h5py.File(fn, "r")
    dg = f[gn]

    param = {"%s" % attr: dg.attrs[attr] for attr in dg.attrs.keys()}
    if isinstance(threads, int):
        param["threads"] = threads

    universe_obj = universe_class(scale_off=True, **param)

    r, v, m, n = dg["r"], dg["v"], dg["m"], dg["name"]
    MASS_LEN = r.shape[0]

    for index in range(MASS_LEN):
        universe_obj.add_object(
            scale_off=True,
            name=bytes(n[index]).decode("utf-8"),
            r=[float(i) for i in r[index, :]],
            v=[float(i) for i in v[index, :]],
            m=float(m[index]),
        )

    f.close()

    return universe_obj


def store_simulation(universe_obj, fn, gn):
    """stores simulation state into HDF5 file"""

    f = h5py.File(fn, "a")
    dg = f.create_group(gn)

    try:
        SIM_DIM = universe_obj.SIM_DIM
    except AttributeError:
        SIM_DIM = len(universe_obj._mass_list[0]._r)
    MASS_LEN = len(universe_obj)
    dtype = {"float32": "<f4", "float64": "<f8"}[universe_obj._dtype]

    names = []
    r = dg.create_dataset("r", (MASS_LEN, SIM_DIM), dtype=dtype)
    v = dg.create_dataset("v", (MASS_LEN, SIM_DIM), dtype=dtype)
    m = dg.create_dataset("m", (MASS_LEN,), dtype=dtype)

    for index, mass_obj in enumerate(universe_obj):
        names.append(mass_obj._name.encode("utf-8"))
        r[index, :] = mass_obj._r[:]
        v[index, :] = mass_obj._v[:]
        m[index] = mass_obj._m

    name_max = max((len(name) for name in names))
    _n = np.chararray((MASS_LEN,), unicode=False, itemsize=name_max)
    for index, name in enumerate(names):
        _n[index] = name
    n = dg.create_dataset("name", (MASS_LEN,), dtype=_n.dtype)
    n[:] = _n[:]

    for attr in [
        "_scale_m",
        "_scale_r",
        "_t",
        "_T",
        "_G",
        "_dtype",
        "_threads",
    ]:
        dg.attrs[attr[1:]] = getattr(universe_obj, attr)

    f.close()
