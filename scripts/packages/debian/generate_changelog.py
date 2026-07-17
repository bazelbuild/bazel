#!/usr/bin/env python3
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
"""This tool generates a debian changelog file."""

from datetime import datetime
import sys


def main(input_file, output_file):
  with open(output_file, "w") as changelog_out:
    version = None
    with open(input_file, "r") as status_file:
      for line in status_file:
        line = line.strip()
        if line.startswith("RELEASE_NAME "):
          version = line[len("RELEASE_NAME "):].strip()

    if version:
      changelog_out.write("bazel (%s) unstable; urgency=low\n" % version)
      changelog_out.write("\n    Bumped Bazel version to %s.\n" % version)
    else:
      changelog_out.write("bazel (0.1.0~HEAD) unstable; urgency=low\n")
      changelog_out.write("\n    Development version\n")
    changelog_out.write(
        "\n -- The Bazel Authors <bazel-discuss@googlegroups.com>  %s\n\n"
        % datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0100")
    )


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
