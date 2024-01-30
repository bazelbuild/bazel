# pylint: disable=g-direct-third-party-import
# Copyright 2021 The Bazel Authors. All rights reserved.
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
"""A tool for extracting the proguard spec file from an AAR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile

# Do not edit this line. Copybara replaces it with PY2 migration helper.
from absl import app
from absl import flags

from tools.android import json_worker_wrapper
from tools.android import junction

FLAGS = flags.FLAGS

flags.DEFINE_string("input_aar", None, "Input AAR")
flags.mark_flag_as_required("input_aar")
flags.DEFINE_string("output_proguard_file", None,
                    "Output parameter file for proguard")
flags.mark_flag_as_required("output_proguard_file")


# Attempt to extract proguard spec from AAR. If the file doesn't exist, an empty
# proguard spec file will be created
def ExtractEmbeddedProguard(aar, output):
  proguard_spec = "proguard.txt"

  if proguard_spec in aar.namelist():
    output.write(aar.read(proguard_spec))


def _Main(input_aar, output_proguard_file):
  with zipfile.ZipFile(input_aar, "r") as aar:
    with open(output_proguard_file, "wb") as output:
      ExtractEmbeddedProguard(aar, output)


def main(unused_argv):
  if os.name == "nt":
    # Shorten paths unconditionally, because the extracted paths in
    # ExtractEmbeddedJars (which we cannot yet predict, because they depend on
    # the names of the Zip entries) may be longer than MAX_PATH.
    aar_long = os.path.abspath(FLAGS.input_aar)
    proguard_long = os.path.abspath(FLAGS.output_proguard_file)

    with junction.TempJunction(os.path.dirname(aar_long)) as aar_junc:
      with junction.TempJunction(
          os.path.dirname(proguard_long)) as proguard_junc:
        _Main(
            os.path.join(aar_junc, os.path.basename(aar_long)),
            os.path.join(proguard_junc, os.path.basename(proguard_long)))
  else:
    _Main(FLAGS.input_aar, FLAGS.output_proguard_file)


if __name__ == "__main__":
  json_worker_wrapper.wrap_worker(FLAGS, main, app.run)
