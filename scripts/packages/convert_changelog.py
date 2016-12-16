# pylint: disable=g-bad-file-header
# Copyright 2015 The Bazel Authors. All rights reserved.
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
"""This tool convert Bazel's changelog file to debian changelog format."""

from datetime import datetime
import sys


def main(input_file, output_file):
  changelog_out = open(output_file, "w")
  time_stamp = None
  with open(input_file, "r") as changelog_in:
    for line in changelog_in:
      line = line.strip()

      if line.startswith("```") or line.startswith("Baseline"):
        continue

      if line.startswith("## Release"):
        if time_stamp:
          changelog_out.write(
              "\n -- The Bazel Authors <bazel-dev@googlegroups.com>  %s\n\n" %
              time_stamp)
        parts = line.split(" ")
        version = parts[2]
        time_stamp = datetime.strptime(
            parts[3], "(%Y-%m-%d)").strftime("%a, %d %b %Y %H:%M:%S +0100")
        changelog_out.write("bazel (%s) unstable; urgency=low\n" % version)

      elif line.startswith("+") or line.startswith("-"):
        parts = line.split(" ")
        line = "*" + line[1:]
        changelog_out.write("  %s\n" % line)

      elif line.endswith(":"):
        changelog_out.write("\n  %s\n" % line)

      elif line:
        changelog_out.write("    %s\n" % line)

    if time_stamp:
      changelog_out.write(
          "\n -- The Bazel Authors <bazel-dev@googlegroups.com>  %s\n\n" %
          time_stamp)


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
