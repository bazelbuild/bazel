# Copyright 2022 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Credential helper for testing."""

import json
import sys


def eprint(*args, **kargs):
    print(*args, file=sys.stderr, **kargs)


def main(argv):
    if len(argv) != 2:
        eprint("Usage: test_credential_helper <command>")
        return 1

    if argv[1] == "get":
        request = json.load(sys.stdin)
        if request['uri'] == "https://singleheader.example.com":
            response = {
                "headers": {
                    "header1": ["value1"],
                },
            }
        elif request['uri'] == "https://multipleheaders.example.com":
            response = {
                "headers": {
                    "header1": ["value1"],
                    "header2": ["value1", "value2"],
                    "header3": ["value1", "value2", "value3"],
                },
            }
        elif request['uri'] == "https://extrafields.example.com":
            response = {
                "foo": "YES",
                "headers": {
                    "header1": ["value1"],
                },
                "umlaut": [
                    "ß",
                    "å",
                ],
            }
        elif request['uri'] == "https://printnothing.example.com":
            return 0
        else:
            eprint("Unknown uri '{}'".format(request['uri']))
            return 1
        json.dump(response, sys.stdout)
        return 0
    else:
        eprint("Unknown command '{}'".format(argv[1]))
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))
