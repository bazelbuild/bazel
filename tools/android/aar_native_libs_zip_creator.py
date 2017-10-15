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

"""A tool for extracting native libs from an AAR into a zip.

The native libs for the requested cpu will be extracted into a zip. The paths
are converted from the AAR directory structure of /jni/<cpu>/foo.so to the APK
directory structure of /lib/<cpu>/foo.so.
"""

import re
import sys
import zipfile

from third_party.py import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_string("input_aar", None, "Input AAR")
gflags.MarkFlagAsRequired("input_aar")
gflags.DEFINE_string("cpu", None, "CPU architecture to include")
gflags.MarkFlagAsRequired("cpu")
gflags.DEFINE_string("output_zip", None, "Output ZIP of native libs")
gflags.MarkFlagAsRequired("output_zip")


class UnsupportedArchitectureException(Exception):
  """Exception thrown when an AAR does not support the requested CPU.
  """
  pass


def CreateNativeLibsZip(aar, cpu, native_libs_zip):
  native_lib_pattern = re.compile("^jni/.+/.+\\.so$")
  if any(native_lib_pattern.match(filename) for filename in aar.namelist()):
    cpu_pattern = re.compile("^jni/" + cpu + "/.+\\.so$")
    libs = [name for name in aar.namelist() if cpu_pattern.match(name)]
    if not libs:
      raise UnsupportedArchitectureException()
    for lib in libs:
      # Only replaces the first instance of jni, in case the AAR contains
      # something like /jni/x86/jni.so.
      new_filename = lib.replace("jni", "lib", 1)
      native_libs_zip.writestr(new_filename, aar.read(lib))


def main():
  with zipfile.ZipFile(FLAGS.input_aar, "r") as input_aar:
    with zipfile.ZipFile(FLAGS.output_zip, "w") as native_libs_zip:
      try:
        CreateNativeLibsZip(input_aar, FLAGS.cpu, native_libs_zip)
      except UnsupportedArchitectureException:
        print ("AAR " + FLAGS.input_aar +
               " missing native libs for requested architecture: " + FLAGS.cpu)
        sys.exit(1)


if __name__ == "__main__":
  FLAGS(sys.argv)
  main()
