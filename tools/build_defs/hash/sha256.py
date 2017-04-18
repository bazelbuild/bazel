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
"""A wrapper to have a portable SHA-256 tool."""

# TODO(dmarting): instead of this tool we should make SHA-256 of artifacts
# available in Skylark.
import hashlib
import sys

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print "Usage: %s input output" % sys.argv[0]
    sys.exit(-1)
  with open(sys.argv[2], "w") as outputfile:
    with open(sys.argv[1], "rb") as inputfile:
      outputfile.write(hashlib.sha256(inputfile.read()).hexdigest())
