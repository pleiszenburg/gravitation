# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/__init__.py: Module init file

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
# PACKAGE META
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__author__ = "Sebastian M. Ernst"
__email__ = "ernst@pleiszenburg.de"
__version__ = "0.2.0"

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EXPORTS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from .lib.baseuniverse import BaseUniverse
from .lib.baseviewer import BaseViewer
from .lib.const import (
    State,
    DIMS,
    Dtype,
    Target,
    Threads,
    DEFAULT_DTYPE,
    DEFAULT_TARGET,
    DEFAULT_THREADS
)
from .lib.debug import (
    DEBUG,
    typechecked,
)
from .lib.errors import (
    BenchmarkLogError,
    KernelError,
    TimerError,
    UniverseError,
    VariationError,
    ViewError,
)
from .lib.kernel import (
    Kernel,
    KERNELS,
)
from .lib.mass import PointMass
from .lib.shm import (
    ShmPool,
    Param,
    WorkerBase,
)
from .lib.timing import (
    BestRunTimer,
    AverageTimer,
    ElapsedTimer,
)
from .lib.variation import (
    Option,
    Variation,
    Variations,
)
from .lib.view import (
    View,
    VIEWS,
)
