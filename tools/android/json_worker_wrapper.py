# pylint: disable=g-direct-third-party-import
# Copyright 2023 The Bazel Authors. All rights reserved.
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
"""A wrapper for AAR extractors to support persistent worker."""

import json
import sys


def wrap_worker(flags, main, app_run):
  """When the --persistent_worker flag is set, it runs in worker mode, otherwise it runs the main method directly.

  Args:
    flags: The FlagValues instance to parse command line arguments.
    main: The main function to execute.
    app_run: app.run() function in Abseil.
  """
  if sys.argv[1] == "--persistent_worker":
    if "--run_with_pdb" in sys.argv or "--run_with_profiling" in sys.argv:
      sys.stderr.write("pdb and profiling are not supported in worker mode.\n")
      sys.exit(1)
    while True:
      # parse work requests from stdin
      request_str = sys.stdin.readline()
      work_request = json.loads(request_str)

      args = [sys.argv[0]] + work_request["arguments"]
      flags(args)
      _wrap_result(main, args)
  else:
    flags(sys.argv)
    app_run(main)


def _wrap_result(main, args):
  """Execute main function and return worker response.

  Args:
    main: The main function to execute.
    args: A non-empty list of the command line arguments including program name.
  """
  response = {
      "exitCode": 0,
      "output": "",
  }
  main(args)
  sys.stdout.write(json.dumps(response))
  sys.stdout.flush()
