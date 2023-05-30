# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/lib/proc.py: Subprocess infrastructure

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

from io import BufferedReader
import os
from queue import Empty, Queue
from subprocess import Popen, PIPE
from threading import Thread
from time import sleep
from typing import Callable, List, Optional, Tuple

from .const import Stream
from .debug import typechecked

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
class _Reader:
    "read from stream in thread"

    def __init__(self, stream_id: Stream, stream: BufferedReader, processing: Callable):
        self._stream_id = stream_id
        self._stream = stream
        self._processing = processing
        self._output = []
        self._queue = Queue()
        self._thread = Thread(target=self._worker)
        self._thread.daemon = True
        self._thread.start()

    def _worker(self):
        "thread worker function"
        for line in iter(self._stream.readline, b""):
            self._queue.put(line)
        self._stream.close()

    @property
    def output(self) -> str:
        "stream output as single string"
        return "".join(self._output)

    def read(self):
        "read from stream in main thread"
        while not self._queue.empty():
            try:
                line = self._queue.get_nowait()
            except Empty:
                pass
            else:
                line = line.decode("utf-8")
                self._output.append(line)
                self._processing(self._stream_id, line.strip("\n"))
                self._queue.task_done()

    def close(self):
        "join queue and thread"
        self._queue.join()
        self._thread.join()


@typechecked
def run_command(
    cmd: List[str], unbuffer: bool = False, processing: Optional[Callable] = None
) -> Tuple[bool, str, str]:
    "subprocess.Popen wrapper, reads stdout and stderr in realtime"

    os.environ["PYTHONUNBUFFERED"] = "1" if unbuffer else 0
    if processing is None:
        processing = print

    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout = _Reader(stream_id=Stream.stdout, stream=proc.stdout, processing=processing)
    stderr = _Reader(stream_id=Stream.stderr, stream=proc.stderr, processing=processing)

    while True:
        sleep(0.2)
        stdout.read()
        stderr.read()
        if proc.poll() is not None:
            break

    stdout.close()
    stderr.close()

    return (not bool(proc.returncode), stdout.output, stderr.output)
