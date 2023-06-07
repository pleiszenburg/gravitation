# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/view/pygame/view.py: pygame view backend

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

from typing import Tuple

import pygame

from . import DESCRIPTION

from gravitation import typechecked
from gravitation import BaseViewer

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class Viewer(BaseViewer):
    __doc__ = DESCRIPTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        unit = 1e20,  # TODO compute from universe?
        unit_size = (16.0, 10.0),  # TODO compute from universe?
        base = 1024  # TODO parameter?

        pygame.init()

        self._font_size = 20
        self._font = pygame.font.SysFont("Consolas", self._font_size)
        self._font_color = (0, 0, 255)
        self._bg_color = (0, 0, 0)
        self._mass_color = (255, 255, 0)
        self._reference_linewidth = 2
        self._reference_color = (255, 0, 0)
        self._unit = unit * self._universe.scale_r
        self._scale_factor = base / self._unit / unit_size[0]
        self._pixel_size = [base, int(base * unit_size[1] / unit_size[0])]
        self._pixel_null = [dim // 2 for dim in self._pixel_size]
        self._canvas = pygame.display.set_mode(self._pixel_size)

    def _draw_text(self, text: str, pos: Tuple[float, float], right: bool = False):
        "draw text onto canvas"

        rendered_text = self._font.render(text, True, self._font_color)
        if right:
            self._canvas.blit(
                rendered_text, (pos[0] - rendered_text.get_width(), pos[1])
            )
        else:
            self._canvas.blit(rendered_text, pos)

    def _render_frame(self):
        "renders one frame"

        self._canvas.fill(self._bg_color)

        pygame.draw.circle(
            self._canvas,
            self._reference_color,
            self._pixel_null,
            int(self._unit * self._scale_factor),
            self._reference_linewidth,
        )

        for mass in self._universe:
            p = [
                int(sign * pos * self._scale_factor + off)
                for sign, pos, off in zip([1, -1], mass.r, self._pixel_null)
            ]
            pygame.draw.circle(self._canvas, self._mass_color, p, 5)

        self._draw_text(
            f"{1.0e9 / self._timer_fps.avg:.02f} F/s",
            (self._pixel_size[0], 0),
            right=True,
        )
        self._draw_text(
            f"{1.0e9 / self._timer_sps.avg:.02f} S/s",
            (self._pixel_size[0], self._font_size),
            right=True,
        )
        self._draw_text(
            f"{self._timer_sps.avg * 1.0e-9:01e} s/S",
            (self._pixel_size[0], 2 * self._font_size),
            right=True,
        )

        pygame.display.flip()

    def _handle_events(self):
        "handles events"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._exit()

    def run(self):
        "runs viewer"

        while True:
            self._simulation()
            self._handle_events()
            self._render_frame()
