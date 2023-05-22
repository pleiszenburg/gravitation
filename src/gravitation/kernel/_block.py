# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/_block.py: Infrastructure for parallelization

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

from typing import Any, List

from typeguard import typechecked

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@typechecked
class Block:
    "Block of indices from start to stop covering n pairs of point masses"

    def __init__(self, n: int, start: int, stop: int):
        self._n = n
        self._start = start
        self._stop = stop

    def __repr__(self) -> str:
        return f'<Block n={self._n:d} start={self._start:d} stop={self._stop:d}>'

    def __add__(self, other: Any):
        if other is None:
            return self.copy()
        assert isinstance(other, type(self))
        return type(self)(
            n = self.n + other.n,
            start = self.start,
            stop = other.stop,
        )

    def __radd__(self, other: Any):
        return self.__add__(other)

    @property
    def n(self) -> int:
        return self._n

    @property
    def start(self) -> int:
        return self._start

    @property
    def stop(self) -> int:
        return self._stop

    def copy(self):
        return type(self)(
            n = self.n,
            start = self.start,
            stop = self.stop,
        )

    @classmethod
    def get_blocks(cls, n: int, vec: int = 1) -> List:
        "Get all vector blocks for a given width of vector register"

        assert n > 0
        assert vec > 0

        blocks = []

        start = 0
        block_idx = 0
        block_n = 0
        for idx in range(0, n - 1):

            block_n += n - idx - 1
            block_idx += 1

            if block_idx == vec:
                blocks.append(cls(
                    n = block_n,
                    start = start,
                    stop = start + block_idx,
                ))
                block_n = 0
                block_idx = 0
                start += vec

        if block_n > 0:
            blocks.append(cls(
                n = block_n,
                start = start,
                stop = start + block_idx,
            ))

        return blocks

    @classmethod
    def get_segments(cls, n: int, threads: int = 2, vec: int = 1) -> List:
        "Get roughly as many groups of blocks, segments, as there are threads"

        assert n > 0
        assert threads > 0
        assert vec > 0

        blocks = cls.get_blocks(n = n, vec = vec)

        if threads >= len(blocks):
            return blocks

        pairs_total = (n * (n - 1)) // 2
        pairs_per_thread = pairs_total // threads

        assert threads * pairs_per_thread <= pairs_total

        segments = []

        current_segment = None
        for block in blocks:
            current_segment = current_segment + block
            if current_segment.n > pairs_per_thread:
                segments.append(current_segment)
                current_segment = None

        if current_segment is not None and current_segment.n > 0:
            segments.append(current_segment)

        assert len(segments) <= threads
        assert sum(segment.n for segment in segments) == pairs_total

        return segments
