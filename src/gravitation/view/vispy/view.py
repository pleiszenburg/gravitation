# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/view/vispy/view.py: vispy view backend

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

from . import DESCRIPTION

from gravitation import typechecked
from gravitation import BaseViewer

import numpy as np
from vispy import app
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Markers, XYZAxis

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@typechecked
class Viewer(BaseViewer):
    __doc__ = DESCRIPTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._canvas = SceneCanvas(keys='interactive', show=True)
        self._view = self._canvas.central_widget.add_view()

        self._pos = np.zeros((len(self._universe), 3), dtype = 'f4')
        self._push_data()
        symbols = {
            "back hole": "x",
            "cloud star": "^",
            "disk star": "o",
        }
        self._symbols = np.array([
            symbols[mass.name]
            for mass in self._universe
        ])

        self._scatter = Markers(
            pos = self._pos,
            edge_width=0,
            face_color=(1, 1, 1, .5),
            size=5,
            symbol=self._symbols,
            parent=self._view.scene,
        )

        self._view.camera = 'turntable'  # alternative: 'turntable / arcball'
        self._view.camera.distance = 10.0

        self._axis = XYZAxis(parent=self._view.scene)
        self._timer = app.Timer('auto', connect = self._update, start = True)

    def _push_data(self):

        for idx, mass in enumerate(self._universe):
            self._pos[idx, :] = [n * 1e-10 for n in mass.r]

    def _update(self, event):

        self._simulation()
        self._push_data()

        self._scatter.set_data(
            pos = self._pos,
            edge_width=0,
            face_color=(1, 1, 1, .5),
            size=5,
            symbol=self._symbols,
        )

    def run(self):
        "runs viewer"

        self._canvas.show()
        app.run()
