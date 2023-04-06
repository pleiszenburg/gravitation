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

from typeguard import typechecked

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONST
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

STDOUT = 1
STDERR = 2

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
def _read_stream_worker(in_stream: BufferedReader, out_queue: Queue):
    """reads lines from stream and puts them into queue"""
    for line in iter(in_stream.readline, b""):
        out_queue.put(line)
    in_stream.close()


@typechecked
def _start_reader(in_stream: BufferedReader) -> Tuple[Thread, Queue]:
    """starts reader thread and returns a thread object and a queue object"""
    out_queue = Queue()
    reader_thread = Thread(
        target=_read_stream_worker, args=(in_stream, out_queue)
    )
    reader_thread.daemon = True
    reader_thread.start()
    return reader_thread, out_queue


@typechecked
def _read_stream(stream_id: int, in_queue: Queue, out_list: List[str], processing: Callable):
    """reads lines from queue and processes them"""
    try:
        line = in_queue.get_nowait()
    except Empty:
        pass
    else:
        line = line.decode("utf-8")
        out_list.append(line)
        processing(stream_id, line.strip("\n"))
        in_queue.task_done()


@typechecked
def run_command(cmd: List[str], unbuffer: bool = False, processing: Optional[Callable] = None) -> Tuple[bool, str, str]:
    """subprocess.Popen wrapper, reads stdout and stderr in realtime"""
    if unbuffer:
        os.environ["PYTHONUNBUFFERED"] = "1"
    else:
        os.environ["PYTHONUNBUFFERED"] = "0"
    if processing is None:
        processing = print
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout_thread, stdout_queue = _start_reader(proc.stdout)
    stderr_thread, stderr_queue = _start_reader(proc.stderr)
    stdout_list, stderr_list = [], []
    while True:
        sleep(0.2)
        while not stdout_queue.empty():
            _read_stream(STDOUT, stdout_queue, stdout_list, processing)
        while not stderr_queue.empty():
            _read_stream(STDERR, stderr_queue, stderr_list, processing)
        if proc.poll() is not None:
            break
    stdout_queue.join()
    stderr_queue.join()
    stdout_thread.join()
    stderr_thread.join()
    return (not bool(proc.returncode), "".join(stdout_list), "".join(stderr_list))
