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
"""A simple cross-platform helper to remove top from a tar file."""
import sys

from tools.build_defs.pkg import archive

if __name__ == '__main__':
  with archive.TarFileWriter(sys.argv[2]) as f:
    f.add_tar(sys.argv[1], name_filter=lambda x: not x.endswith('top'))
