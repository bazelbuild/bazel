# Lint as: python2, python3
# pylint: disable=g-direct-third-party-import
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import zipfile

# Do not edit this line. Copybara replaces it with PY2 migration helper.
from absl import app
from absl import flags
import six

from tools.android import junction

FLAGS = flags.FLAGS

flags.DEFINE_string("input_aar", None, "Input AAR")
flags.mark_flag_as_required("input_aar")
flags.DEFINE_string("cpu", None, "CPU architecture to include")
flags.mark_flag_as_required("cpu")
flags.DEFINE_string("output_zip", None, "Output ZIP of native libs")
flags.mark_flag_as_required("output_zip")


class UnsupportedArchitectureException(Exception):
  """Exception thrown when an AAR does not support the requested CPU."""
  pass


def CreateNativeLibsZip(aar, cpu, native_libs_zip):
  native_lib_pattern = re.compile("^jni/.+/.+\\.so$")
  if any(native_lib_pattern.match(filename) for filename in aar.namelist()):
    cpu_pattern = re.compile("^jni/" + six.ensure_str(cpu) + "/.+\\.so$")
    libs = [name for name in aar.namelist() if cpu_pattern.match(name)]
    if not libs:
      raise UnsupportedArchitectureException()
    for lib in libs:
      # Only replaces the first instance of jni, in case the AAR contains
      # something like /jni/x86/jni.so.
      new_filename = lib.replace("jni", "lib", 1)
      native_libs_zip.writestr(new_filename, aar.read(lib))


def Main(input_aar_path, output_zip_path, cpu, input_aar_path_for_error_msg):
  with zipfile.ZipFile(input_aar_path, "r") as input_aar:
    with zipfile.ZipFile(output_zip_path, "w") as native_libs_zip:
      try:
        CreateNativeLibsZip(input_aar, cpu, native_libs_zip)
      except UnsupportedArchitectureException:
        print("AAR " + six.ensure_str(input_aar_path_for_error_msg) +
              " missing native libs for requested architecture: " +
              six.ensure_str(cpu))
        sys.exit(1)


def main(unused_argv):
  if os.name == "nt":
    with junction.TempJunction(os.path.dirname(FLAGS.input_aar)) as j_in:
      with junction.TempJunction(os.path.dirname(FLAGS.output_zip)) as j_out:
        Main(
            os.path.join(j_in, os.path.basename(FLAGS.input_aar)),
            os.path.join(j_out, os.path.basename(FLAGS.output_zip)), FLAGS.cpu,
            FLAGS.input_aar)
  else:
    Main(FLAGS.input_aar, FLAGS.output_zip, FLAGS.cpu, FLAGS.input_aar)


if __name__ == "__main__":
  FLAGS(sys.argv)
  app.run(main)
