# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/lib/option.py: CLI options from variations and platform information

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

from enum import Enum
from typing import Any, Tuple, Type

from .debug import typechecked

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

    @property
    def name(self) -> str:
        "name of option"
        return self._name

    @property
    def type(self) -> Type:
        "type of option"
        return self._type

    @property
    def choices(self) -> Tuple[Any]:
        "all choices of option, sorted; stripped if enum"
        return tuple(sorted(
            getattr(choice, 'name', choice)
            for choice in self._choices
        ))

    def add(self, choice: Any):
        "add choice"
        if not isinstance(choice, self._type):
            raise TypeError('wrong choice type for option')
        self._choices.add(choice)

    def value_to_type(self, value: Any) -> Any:
        if issubclass(self._type, Enum):
            return self._type[value]
        return self._type(value)
