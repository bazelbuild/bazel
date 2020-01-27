# Copyright 2018 The Bazel Authors. All rights reserved.
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
"""Mock Python binary, only used in tests."""

from __future__ import print_function
from bazel_tools.tools.python.runfiles import runfiles

print('Hello Python Bar!')
r = runfiles.Create()
print('rloc=%s' % r.Rlocation('foo_ws/bar/bar-py-data.txt'))
