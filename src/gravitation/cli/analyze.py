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

import click

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROUTINES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def _parse_linestr_to_linedict(line_str):
    """parse single line from log (json str) to python dict"""
    try:
        return json.loads(line_str)
    except json.decoder.JSONDecodeError:
        return {"log": "JSON_ERROR", "str": line_str}


def _parse_itemstr_to_itemdict(item_str):
    """parse log of one full benchmark worker run to one single dict"""

    item_dict = {}

    line_list = [
        _parse_linestr_to_linedict(line)
        for line in item_str.split("\n")
        if line.strip() != ""
    ]

    errors = [line_dict for line_dict in line_list if line_dict["log"] == "JSON_ERROR"]
    if len(errors) != 0:
        print(errors)
        raise SyntaxError("benchmark log has non-JSON components, likely errors")

    errors = [line_dict for line_dict in line_list if line_dict["log"] == "ERROR"]
    if len(errors) != 0:
        print(errors)
        raise SyntaxError("benchmark has errors")

    input = [line_dict for line_dict in line_list if line_dict["log"] == "INPUT"]
    if len(input) > 1:
        raise SyntaxError("more than one INPUT log per benchmark worker run")
    if len(input) < 1:
        raise SyntaxError("INPUT log missing in benchmark worker run")
    input = copy.deepcopy(input)
    input[0].pop("log")
    item_dict["meta"] = input[0]

    size = copy.deepcopy(
        [line_dict for line_dict in line_list if line_dict["log"] == "SIZE"]
    )
    if len(size) > 1:
        raise SyntaxError("more than one SIZE log per benchmark worker run")
    if len(size) < 1:
        raise SyntaxError("SIZE log missing in benchmark worker run")
    item_dict["meta"]["simulation"]["size"] = size[0]["value"]

    item_dict["runtime"] = [
        line_dict["runtime"] for line_dict in line_list if line_dict["log"] == "STEP"
    ]
    item_dict["gctime"] = [
        line_dict["gctime"] for line_dict in line_list if line_dict["log"] == "STEP"
    ]

    counter = [
        line_dict["counter"] for line_dict in line_list if line_dict["log"] == "STEP"
    ]
    if len(counter) == 0:
        raise SyntaxError("benchmark did not run any steps")
    if counter != list(range(counter[0], len(counter) + counter[0])):
        raise SyntaxError("benchmark has unexpected sequence of steps")

    if line_list[-1] != {"log": "EXIT", "msg": "OK"}:
        raise SyntaxError("benchmark did not exit properly")

    return item_dict


def _parse_logstr_to_datalist(log_str):
    """parse a benchmark log consisting of multiple worker runs to list of dict"""
    return [
        _parse_itemstr_to_itemdict(item)
        for item in log_str.split('{"log": "START"}\n')
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
