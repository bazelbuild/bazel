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

"""A tool for extracting all jar files from an AAR.

An AAR may contain JARs at /classes.jar and /libs/*.jar. This tool extracts all
of the jars and creates a param file for singlejar to merge them into one jar.
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
flags.DEFINE_string("output_singlejar_param_file", None,
                    "Output parameter file for singlejar")
flags.mark_flag_as_required("output_singlejar_param_file")
flags.DEFINE_string("output_dir", None, "Output directory to extract jars in")
flags.mark_flag_as_required("output_dir")


def ExtractEmbeddedJars(aar,
                        singlejar_param_file,
                        output_dir,
                        output_dir_orig=None):
  if not output_dir_orig:
    output_dir_orig = output_dir
  jar_pattern = re.compile("^(classes|libs/.+)\\.jar$")
  singlejar_param_file.write(b"--exclude_build_data\n")
  for name in aar.namelist():
    if jar_pattern.match(name):
      singlejar_param_file.write(b"--sources\n")
      # output_dir may be a temporary junction, so write the original
      # (unshortened) path to the params file
      singlejar_param_file.write(
          six.ensure_binary((output_dir_orig + "/" + name + "\n"), "utf-8"))
      aar.extract(name, output_dir)


def _Main(input_aar,
          output_singlejar_param_file,
          output_dir,
          output_dir_orig=None):
  if not output_dir_orig:
    output_dir_orig = output_dir
  with zipfile.ZipFile(input_aar, "r") as aar:
    with open(output_singlejar_param_file, "wb") as singlejar_param_file:
      ExtractEmbeddedJars(aar, singlejar_param_file, output_dir,
                          output_dir_orig)


def main(unused_argv):
  if os.name == "nt":
    # Shorten paths unconditionally, because the extracted paths in
    # ExtractEmbeddedJars (which we cannot yet predict, because they depend on
    # the names of the Zip entries) may be longer than MAX_PATH.
    aar_long = os.path.abspath(FLAGS.input_aar)
    params_long = os.path.abspath(FLAGS.output_singlejar_param_file)
    out_long = os.path.abspath(FLAGS.output_dir)
    with junction.TempJunction(os.path.dirname(aar_long)) as aar_junc:
      with junction.TempJunction(os.path.dirname(params_long)) as params_junc:
        with junction.TempJunction(os.path.dirname(out_long)) as out_junc:
          _Main(
              os.path.join(aar_junc, os.path.basename(aar_long)),
              os.path.join(params_junc, os.path.basename(params_long)),
              os.path.join(out_junc, os.path.basename(out_long)),
              FLAGS.output_dir)
  else:
    _Main(FLAGS.input_aar, FLAGS.output_singlejar_param_file, FLAGS.output_dir)


if __name__ == "__main__":
  FLAGS(sys.argv)
  app.run(main)
