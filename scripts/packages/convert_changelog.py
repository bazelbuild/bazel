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
  changelog = None
  version = None
  with open(input_file, "r") as changelog_in:
    for line in changelog_in:
      line = line.strip()

      if line.startswith("RELEASE_NOTES"):
        changelog = line[14:].strip().replace("\f", "\n")
      elif line.startswith("RELEASE_NAME"):
        version = line[13:].strip()

  if changelog:
    time_stamp = None
    lines = changelog.split("\n")
    header = lines[0]
    lines = lines[4:]  # Skip the header
    if lines[0] == "Cherry picks:":
      # Skip cherry picks list
      i = 1
      while lines[i].strip():
        i += 1
      lines = lines[i + 1:]

    # Process header
    parts = header.split(" ")
    time_stamp = datetime.strptime(
        parts[2], "(%Y-%m-%d)").strftime("%a, %d %b %Y %H:%M:%S +0100")
    changelog_out.write("bazel (%s) unstable; urgency=low\n" % version)

    for line in lines:
      if line.startswith("+") or line.startswith("-"):
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
