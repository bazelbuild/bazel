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

import argparse
import os
import re
import sys
import zipfile

from tools.android import junction

class UnsupportedArchitectureException(Exception):
  """Exception thrown when an AAR does not support the requested CPU."""
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
      # To guarantee reproducible zips we must specify a new zipinfo.
      # From writestr docs: "If its a name, the date and time is set to the
      # current date and time." which will break the HASH calculation and result
      # in a cache miss.
      old_zipinfo = aar.getinfo(lib)
      new_zipinfo = zipfile.ZipInfo(filename=new_filename)
      new_zipinfo.date_time = old_zipinfo.date_time
      new_zipinfo.compress_type = old_zipinfo.compress_type

      native_libs_zip.writestr(new_zipinfo, aar.read(lib))


def Main(input_aar_path, output_zip_path, cpu, input_aar_path_for_error_msg):
  with zipfile.ZipFile(input_aar_path, "r") as input_aar:
    with zipfile.ZipFile(output_zip_path, "w") as native_libs_zip:
      try:
        CreateNativeLibsZip(input_aar, cpu, native_libs_zip)
      except UnsupportedArchitectureException:
        print("AAR " + input_aar_path_for_error_msg +
              " missing native libs for requested architecture: " +
              cpu)
        sys.exit(1)


def main():
  parser = argparse.ArgumentParser(description='A tool for extracting native libs from an AAR into a zip.')
  parser.add_argument('--input_aar', required=True, help='Input AAR')
  parser.add_argument('--cpu', required=True, help='CPU architecture to include')
  parser.add_argument('--output_zip', required=True, help='Output ZIP of native libs')
  args = parser.parse_args()
  if os.name == "nt":
    with junction.TempJunction(os.path.dirname(args.input_aar)) as j_in:
      with junction.TempJunction(os.path.dirname(args.output_zip)) as j_out:
        Main(
            os.path.join(j_in, os.path.basename(args.input_aar)),
            os.path.join(j_out, os.path.basename(args.output_zip)), args.cpu,
            args.input_aar)
  else:
    Main(args.input_aar, args.output_zip, args.cpu, args.input_aar)


if __name__ == "__main__":
  main()
