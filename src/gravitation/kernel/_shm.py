# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/_shm.py: Shared memory infrastructure

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

from abc import ABC
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Process, Queue
from random import randint
import socket
from typing import Generator, List, Tuple, Union

import numpy as np

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASSES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class Worker(ABC):
    "Base class for worker with shared memory support"

    def __init__(self, idx: int, port: int, authkey: bytes, in_queue: Queue, out_queue: Queue):
        self._idx = idx
        self._mngr = SharedMemoryManager(address = ('localhost', port), authkey = authkey)
        self._mngr.connect()
        self._arrays = {}
        self._buffers = {}
        self._in_queue = in_queue
        self._out_queue = out_queue

    def __getitem__(self, name: str) -> np.ndarray:
        return self._arrays[name]

    def run(self):
        "Mainloop, waiting for and running tasks"

        while True:
            cmd, args, kwargs = self._in_queue.get()
            if cmd == 'ping':
                self._out_queue.put(True)
                continue
            if cmd == 'register':
                dtype = np.dtype(kwargs['dtype'])
                self._buffers[kwargs['name']] = kwargs['buffer']
                self._arrays[kwargs['name']] = np.ndarray(
                    kwargs['shape'],
                    dtype = dtype,
                    order = kwargs['order'],
                    buffer = self._buffers[kwargs['name']].buf,
                )
                self._out_queue.put(True)
                continue
            if cmd == 'close':
                break
            self._out_queue.put(getattr(self, cmd)(*args, **kwargs))

    @classmethod
    def pool_init(cls, *args, **kwargs):
        "Entry point for process"

        worker = cls(*args, **kwargs)
        worker.run()


class Node:
    "Representing a worker in the main process"

    def __init__(self, idx: int, port: int, authkey: bytes, worker: type, **kwargs):
        if not issubclass(worker, Worker):
            raise TypeError('argument "worker" must be a subclass of Worker')
        self._idx = idx
        self._in_queue = Queue()
        self._out_queue = Queue()
        self._proc = Process(
            target = worker.pool_init,
            args = (idx, port, authkey, self._in_queue, self._out_queue),
            kwargs = kwargs,
        )
        self._proc.start()
        self._active = False

    def __repr__(self) -> str:
        return f'<Node idx={self._idx:d} running={self._proc is not None} active={self._active}>'

    def run(self, cmd: str, *args, **kwargs):
        "Run command (method) on worker"

        if self._active:
            raise SystemError('node active')
        if self._proc is None:
            raise SystemError('node closed')
        self._active = True
        self._in_queue.put((cmd, args, kwargs))

    def wait(self):
        "Wait for current command to complete and return its result"

        if not self._active:
            raise SystemError('node not active')
        if self._proc is None:
            raise SystemError('node closed')
        res = self._out_queue.get()
        self._active = False
        return res

    def register(self, name: str, shape: Tuple[int, ...], dtype: type, order: str, buffer):
        "Register shared array on worker"

        self.run('register', name = name, shape = shape, dtype = dtype.base.name, order = order, buffer = buffer)
        return self.wait()

    def ping(self):
        "Ping worker (health status)"

        self.run('ping')
        return self.wait()

    def close(self):
        "Shut down worker"

        if self._active:
            raise SystemError('node active')
        if self._proc is None:
            raise SystemError('node closed')
        self.run('close')
        self._active = False
        self._proc.join()
        self._proc = None


class Param:
    "Thin wrapper for arguments"

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @property
    def args(self) -> tuple:
        return self._args

    @property
    def kwargs(self) -> dict:
        return self._kwargs


class ShmPool:
    "Minimal process pool with integrated shared memory support"

    def __init__(self, nodes: int = 1, worker: type = Worker, **kwargs):
        assert nodes > 0
        port = self.get_free_port()
        authkey = f'{randint(2**50, 2**60):x}'.encode('utf-8')
        self._mngr = SharedMemoryManager(address = ('localhost', port), authkey = authkey)
        self._mngr.start()
        self._nodes = [Node(idx, port, authkey, worker, **kwargs) for idx in range(nodes)]
        self._arrays = {}
        self._buffers = {}

    def __repr__(self) -> str:
        return f'<Manager running={self._nodes is not None} len={len(self):d}>'

    def __len__(self) -> int:
        if self._nodes is None:
            return 0
        return len(self._nodes)

    def __getitem__(self, name: str) -> np.ndarray:
        if self._nodes is None:
            raise SystemError('manager closed')
        return self._arrays[name]

    def __enter__(self):
        if self._nodes is None:
            raise SystemError('manager closed')
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def empty(self, name: str, shape: Tuple[int, ...], dtype: Union[str, type], order: str = 'C') -> np.ndarray:
        "Create and register empty numpy array, similar to np.empty"

        if self._nodes is None:
            raise SystemError('manager closed')
        if name in self._buffers.keys():
            raise ValueError('name already used')
        if any(dim < 0 for dim in shape):
            raise ValueError('negative value in shape')
        dtype = np.dtype(dtype)
        self._buffers[name] = self._mngr.SharedMemory(size = np.prod(shape) * dtype.itemsize)
        self._arrays[name] = np.ndarray(shape, dtype = dtype, order = order, buffer = self._buffers[name].buf)
        _ = [node.register(name, shape, dtype, order, self._buffers[name]) for node in self._nodes]
        return self._arrays[name]

    def keys(self) -> Generator:
        "Expose names of registers arrays"

        return (key for key in self._arrays.keys())

    def run(self, cmd: str, params: List[Param]) -> list:
        "Run command (method) on all workers, distribute parameter sets"

        if len(params) > len(self):
            raise ValueError('more parameter sets than workers')
        _ = [node.run(cmd, *param.args, **param.kwargs) for param, node in zip(params, self._nodes)]
        return [node.wait() for node in self._nodes[:len(params)]]

    def run_all(self, cmd: str, *args, **kwargs):
        "Run command (method) on all workers, identical parameter set"

        _ = [node.run(cmd, *args, **kwargs) for node in self._nodes]
        return [node.wait() for node in self._nodes]

    def ping(self):
        "Ping all workers (health status)"

        if self._nodes is None:
            raise SystemError('manager closed')
        return [node.ping() for node in self._nodes]

    def close(self):
        "Shut down all workers, free shared memory"

        if self._nodes is None:
            raise SystemError('manager closed')
        for node in self._nodes:
            node.close()
        self._arrays = None
        self._buffers = None
        self._nodes = None
        self._mngr.shutdown()
        self._mngr = None

    @staticmethod
    def get_free_port() -> int:
        "Get free port number"

        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port
