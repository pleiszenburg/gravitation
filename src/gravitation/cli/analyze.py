# -*- coding: utf-8 -*-

"""

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/cli/analyze.py: analyze command

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

import copy
import json
from typing import List

import click

from ..lib.debug import typechecked
from ..lib.errors import BenchmarkLogError

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@typechecked
def _parse_linestr_to_linedict(line: str) -> dict:
    """parse single line from log (json str) to python dict"""
    try:
        return json.loads(line)
    except json.decoder.JSONDecodeError:
        return {"log": "JSON_ERROR", "str": line}


@typechecked
def _parse_itemstr_to_itemdict(raw: str) -> dict:
    """parse log of one full benchmark worker run to one single dict"""

    item = {}

    lines = [
        _parse_linestr_to_linedict(line)
        for line in raw.split("\n")
        if line.strip() != ""
    ]

    errors = [line for line in lines if line["log"] == "JSON_ERROR"]
    if len(errors) != 0:
        for idx, error in enumerate(errors):
            print(f"=== ERROR {idx+1:d} ===")
            print(error["str"])
        raise BenchmarkLogError("benchmark log has non-JSON components, likely errors")

    errors = [line for line in lines if line["log"] == "ERROR"]
    if len(errors) != 0:
        print(errors)
        raise BenchmarkLogError("benchmark has errors")

    input_ = [line for line in lines if line["log"] == "INPUT"]
    if len(input_) > 1:
        raise BenchmarkLogError("more than one INPUT log per benchmark worker run")
    if len(input_) < 1:
        raise BenchmarkLogError("INPUT log missing in benchmark worker run")
    input_ = copy.deepcopy(input_)
    input_[0].pop("log")
    item["meta"] = input_[0]

    size = copy.deepcopy([line for line in lines if line["log"] == "SIZE"])
    if len(size) > 1:
        raise BenchmarkLogError("more than one SIZE log per benchmark worker run")
    if len(size) < 1:
        raise BenchmarkLogError("SIZE log missing in benchmark worker run")
    item["meta"]["simulation"]["size"] = size[0]["value"]

    item["runtime"] = [line["runtime"] for line in lines if line["log"] == "STEP"]
    item["gctime"] = [line["gctime"] for line in lines if line["log"] == "STEP"]

    counter = [line["counter"] for line in lines if line["log"] == "STEP"]
    if len(counter) == 0:
        raise BenchmarkLogError("benchmark did not run any steps")
    if counter != list(range(counter[0], len(counter) + counter[0])):
        raise BenchmarkLogError("benchmark has unexpected sequence of steps")

    if lines[-1] != {"log": "EXIT", "msg": "OK"}:
        raise BenchmarkLogError("benchmark did not exit properly")

    return item


@typechecked
def _parse_logstr_to_datalist(log: str) -> List[dict]:
    """parse a benchmark log consisting of multiple worker runs to list of dict"""
    return [
        _parse_itemstr_to_itemdict(item)
        for item in log.split('{"log": "START"}\n')
        if item.strip() != ""
    ]


@click.command(short_help="analyze benchmark logfile")
@click.option(
    "--logfile",
    "-l",
    default="benchmark.log",
    type=click.File("r"),
    show_default=True,
    help="name of input log file",
)
@click.option(
    "--data",
    "-o",
    default="benchmark.json",
    type=click.File("w"),
    show_default=True,
    help="name of output data file",
)
def analyze(logfile, data):
    """analyze benchmark logfile"""
    data.write(
        json.dumps(
            _parse_logstr_to_datalist(logfile.read()),
            indent="\t",
            sort_keys=True,
        )
    )
