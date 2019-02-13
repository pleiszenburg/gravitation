# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/lib/proc.py: Subprocess infrastructure

	Copyright (C) 2019 Sebastian M. Ernst <ernst@pleiszenburg.de>

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

import os
import queue
import subprocess
import threading
import time

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONST
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

STDOUT = 1
STDERR = 2

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _read_stream_worker(in_stream, out_queue):
	"""reads lines from stream and puts them into queue"""
	for line in iter(in_stream.readline, b''):
		out_queue.put(line)
	in_stream.close()

def _start_reader(in_stream):
	"""starts reader thread and returns a thread object and a queue object"""
	out_queue = queue.Queue()
	reader_thread = threading.Thread(
		target = _read_stream_worker, args = (in_stream, out_queue)
		)
	reader_thread.daemon = True
	reader_thread.start()
	return reader_thread, out_queue

def _read_stream(stream_id, in_queue, out_list, processing):
	"""reads lines from queue and processes them"""
	try:
		line = in_queue.get_nowait()
	except queue.Empty:
		pass
	else:
		line = line.decode('utf-8')
		out_list.append(line)
		processing(stream_id, line.strip('\n'))
		in_queue.task_done()

def run_command(cmd_list, unbuffer = False, processing = None):
	"""subprocess.Popen wrapper, reads stdout and stderr in realtime"""
	if unbuffer:
		os.environ['PYTHONUNBUFFERED'] = '1'
	else:
		os.environ['PYTHONUNBUFFERED'] = '0'
	if processing is None:
		processing = print
	proc = subprocess.Popen(cmd_list, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
	stdout_thread, stdout_queue = _start_reader(proc.stdout)
	stderr_thread, stderr_queue = _start_reader(proc.stderr)
	stdout_list, stderr_list = [], []
	proc_alive = True
	while proc_alive:
		time.sleep(0.2)
		while not stdout_queue.empty():
			_read_stream(STDOUT, stdout_queue, stdout_list, processing)
		while not stderr_queue.empty():
			_read_stream(STDERR, stderr_queue, stderr_list, processing)
		proc_alive = proc.poll() is None
	stdout_queue.join()
	stderr_queue.join()
	stdout_thread.join()
	stderr_thread.join()
	return (
		not bool(proc.returncode),
		''.join(stdout_list),
		''.join(stderr_list)
		)
