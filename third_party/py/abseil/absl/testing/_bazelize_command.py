# Copyright 2017 The Abseil Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Internal helper for running tests on Windows Bazel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def get_executable_path(py_binary_path):
  """Returns the executable path of a py_binary.

  This returns the executable path of a py_binary that is in another Bazel
  target's data dependencies.

  On Linux/macOS, it's the same as the py_binary_path.
  On Windows, the py_binary_path points to a zip file, and Bazel 0.5.3+
  generates a .cmd file that can be used to execute the py_binary.

  Args:
    py_binary_path: string, the path of a py_binary that is in another Bazel
        target's data dependencies.
  """
  if os.name == 'nt':
    executable_path = py_binary_path + '.cmd'
    if executable_path.startswith('\\\\?\\'):
      # In Bazel 0.5.3 and Python 3, the paths starts with "\\?\".
      # However, Python subprocess doesn't support those paths well.
      # Strip them as we don't need the prefix.
      # See this page for more informaton about "\\?\":
      # https://msdn.microsoft.com/en-us/library/windows/desktop/aa365247.
      executable_path = executable_path[4:]
    return executable_path
  else:
    return py_binary_path
