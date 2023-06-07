# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/lib/variation.py: Variations and options for individual kernels

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

from json import dumps, loads
from typing import Any, Generator, Optional, Tuple

from .const import (
    Dtype,
    Target,
    Threads,
    DEFAULT_DTYPE,
    DEFAULT_TARGET,
    DEFAULT_THREADS,
)
from .debug import typechecked
from .errors import VariationError

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class Option:
    "Option for use with click"

    def __init__(self, name: str, choice: Any):
        self._name = name
        self._type = type(choice)
        self._choices = {choice}

    def __repr__(self) -> str:
        return f'<Option name={self._name:s} type={self._type.__name__:s} choices={str(self._choices):s}>'

    def __len__(self) -> int:
        return len(self._choices)

    def __contains__(self, choice: Any) -> bool:
        return choice in self._choices

    def add(self, choice: Any):
        if not isinstance(choice, self._type):
            raise TypeError('wrong choice type for option')
        self._choices.add(choice)


@typechecked
class Variation:
    "Variation of how a kernel can operate"

    _ENUMS = (Dtype, Target, Threads)

    def __init__(
        self,
        dtype: Dtype,
        target: Target,
        threads: Threads,
        **meta: Any,
    ):
        self._dtype = dtype
        self._target = target
        self._threads = threads
        self._meta = {k: v for k, v in meta.items() if v is not None}

    def __repr__(self) -> str:
        ret = ['<Variation']
        for enum in self._ENUMS:
            value = getattr(self, f'_{enum.__name__.lower():s}').name
            ret.append(f' {enum.__name__.lower():s}={value}')
        for field, value in self._meta.items():
            ret.append(f' {field:s}="{value}"')
        ret.append('>')
        return ''.join(ret)

    def __len__(self) -> int:
        return len(self._ENUMS) + len(self._meta)

    def __getitem__(self, field: str) -> Any:
        return self._meta[field]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.key == other.key

    @property
    def key(self) -> Tuple[str, ...]:
        return (self[field] for field in self.keys())

    def keys(self) -> Generator:
        yield from (enum.__name__.lower() for enum in self._ENUMS)
        yield from sorted(self._meta.keys())

    def to_json(self) -> str:
        return dumps(self.to_dict(), sort_keys = True)

    def to_dict(self) -> dict:
        return {field: self.getvalue(field) for field in self.keys()}

    def getvalue(self, field: str) -> Any:
        if field not in (enum.__name__.lower() for enum in self._ENUMS):
            return self[field]
        return self[field].name

    @classmethod
    def from_json(cls, raw: str):
        return cls.from_dict(**loads(raw))

    @classmethod
    def from_dict(cls, **kwargs: Any):
        for enum in cls._ENUMS:
            kwargs[enum.__name__.lower()] = enum[kwargs[enum.__name__.lower()]]
        return cls(**kwargs)

    @classmethod
    def from_default(cls):
        return cls(
            dtype = DEFAULT_DTYPE,
            target = DEFAULT_TARGET,
            threads = DEFAULT_THREADS,
        )


@typechecked
class Variations:
    "All variations of how a kernel can operate"

    def __init__(self, *variations: Variation):
        self._variations = {variation.key: variation for variation in variations}
        self._selected = None

    def __repr__(self) -> str:
        return f'<Variations len={len(self):d}>'

    def __len__(self) -> int:
        return len(self._variations)

    def __iter__(self) -> Generator:
        return (variation for variation in self._variations.values())

    def __contains__(self, variation: Optional[Variation]) -> bool:
        if variation is None:
            return False
        return variation.key in self._variations.keys()

    def add(self, variation: Variation):
        self._variations[variation.key] = variation

    def to_options(self) -> Tuple[Option, ...]:
        options = {}
        for variation in self:
            for field in variation.keys():
                if field in options.keys():
                    options[field].add(variation.getvalue(field))
                else:
                    options[field] = Option(field, variation.getvalue(field))
        return tuple(options.values())

    @property
    def selected(self) -> Variation:
        if self._selected is None:
            raise VariationError('no variation selected')
        return self._selected

    @selected.setter
    def selected(self, variation: Variation):
        if variation not in self:
            raise VariationError('variation can not be selected, not part of possible variations')
        self._selected = variation
