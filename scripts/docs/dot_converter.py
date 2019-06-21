# Copyright 2016 The Bazel Authors. All rights reserved.
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
"""Graphviz documentation converter library."""

import fileinput
import re
import subprocess
import sys


class DotConverter(object):
  """Converts dot graphs to SVG format inline."""

  def __init__(self, dot_command, dot_env):
    self.dot_command = dot_command
    self.dot_env = dot_env

  def convert(self):
    collect = False
    graph = b""
    block_num = 0

    for line in fileinput.input():
      if re.search(r"div class='graphviz dot'><!--", line):
        collect = True
        continue
      elif collect and re.search(r"--></div>", line):
        dot = subprocess.Popen(
            self.dot_command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=self.dot_env)
        output = dot.communicate(graph)[0].decode()
        if dot.returncode == 0:
          # cut the first few lines (svg header + comments)
          sys.stdout.write(output.split("\n", 6)[6])
        else:
          sys.stderr.write("inlining block %d failed.\n" % (block_num + 1))
        collect = False
        graph = b""
        block_num += 1
        continue

      if collect:
        graph += str.encode(line)
      else:
        sys.stdout.write(line)
