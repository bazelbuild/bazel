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

import os
import subprocess

from bazel_tools.tools.runfiles import runfiles


def IsWindows():
  return os.name == "nt"


def ChildBinaryName(lang):
  if IsWindows():
    return "foo_ws/bar/bar-%s.exe" % lang
  else:
    return "foo_ws/bar/bar-%s" % lang


def main():
  print("Hello Python Foo!")
  r = runfiles.Create()
  print("rloc=%s" % r.Rlocation("foo_ws/foo/datadep/hello.txt"))

  # Run a subprocess, propagate the runfiles envvar to it. The subprocess will
  # use this process's runfiles manifest or runfiles directory.
  if IsWindows():
    env = {"SYSTEMROOT": os.environ["SYSTEMROOT"]}
  else:
    env = {}
  env.update(r.EnvVar())
  p = subprocess.Popen(
      [r.Rlocation(ChildBinaryName("py"))],
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE)
  for e in p.communicate():
    print(e)


if __name__ == "__main__":
  main()
