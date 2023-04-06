# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/realtimeview/backend_pygame.py: realtimeview pygame backend

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
from typing import List, Optional, Tuple

import pygame
from typeguard import typechecked

from ...lib.load import inventory
from ...lib.simulation import create_simulation
from ...lib.timing import AverageTimer

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class Realtimeview:
    """
    viewer based on pygame
    """

    def __init__(
        self,
        kernel: str,
        threads: int,
        scenario: str,
        scenario_param: dict,
        steps_per_frame: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ):
        self._max_iterations = max_iterations
        self._iteration_counter = 0
        inventory[kernel].load_module()
        self._universe = create_simulation(
            scenario=scenario,
            universe_class=inventory[kernel].get_class(),
            scenario_param=scenario_param,
            threads=threads,
        )
        self._timer_sps = AverageTimer(self._universe._screen["average_over_steps"])
        self._timer_fps = AverageTimer(self._universe._screen["average_over_steps"])
        self._init_canvas(
            spf=self._universe._screen["steps_per_frame"]
            if steps_per_frame is None
            else steps_per_frame,
            unit=self._universe._screen["unit"],
            unit_size=self._universe._screen["unit_size"],
        )

    def _init_canvas(self, spf: int, unit: float, unit_size: List[float], base: int = 1024):
        pygame.init()
        self._font_size = 20
        self._font = pygame.font.SysFont("Consolas", self._font_size)
        self._font_color = (0, 0, 255)
        self._bg_color = (0, 0, 0)
        self._mass_color = (255, 255, 0)
        self._reference_linewidth = 2
        self._reference_color = (255, 0, 0)
        self._unit = unit * self._universe._scale_r
        self._scale_factor = base / self._unit / unit_size[0]
        self._pixel_size = [base, int(base * unit_size[1] / unit_size[0])]
        self._pixel_null = [dim // 2 for dim in self._pixel_size]
        self._canvas = pygame.display.set_mode(self._pixel_size)
        self._spf = spf
        self._timer_sps.start()
        self._timer_sps.stop()
        self._timer_fps.start()

    def _draw_text(self, text: str, pos: Tuple[float, float], right: bool = False):
        rendered_text = self._font.render(text, True, self._font_color)
        if right:
            self._canvas.blit(
                rendered_text, (pos[0] - rendered_text.get_width(), pos[1])
            )
        else:
            self._canvas.blit(rendered_text, pos)

    def _exit(self):
        sys.exit()

    def _loop_canvas(self):
        self._timer_fps.stop()
        self._timer_fps.start()
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
                for sign, pos, off in zip([1, -1], mass._r, self._pixel_null)
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

    def _loop_counter(self):
        self._iteration_counter += 1
        if self._max_iterations is None:
            return
        if self._iteration_counter > self._max_iterations:
            self._exit()

    def _loop_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._exit()

    def _loop_simulation(self):
        for _ in range(self._spf):
            self._timer_sps.start()
            self._universe.step()
            self._timer_sps.stop()

    def loop(self):
        while True:
            self._loop_counter()
            self._loop_events()
            self._loop_canvas()
            self._loop_simulation()
