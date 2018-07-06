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
#
# Convenience macro for skydoc tests. Each target represents two targets:
# 1. A sh_test target which verifies that skydoc, when run on an input file,
#    creates output matching the contents of a golden file.
# 2. A genrule target which will generate a new golden file given an input file
#    and the current version of skydoc. This target should be used to regenerate
#    the golden file if changes are made to skydoc.
"""Convenience macro for skydoc tests."""

def skydoc_test(name, input_file, golden_file, skydoc, deps = []):
    """Creates a test target and golden-file regeneration target for skydoc testing.

    The test target is named "{name}_e2e_test".
    The golden-file regeneration target is named "regenerate_{name}_golden".

    Args:
      name: A unique name to qualify the created targets.
      input_file: The label string of the skylark input file for which documentation is generated
          in this test.
      golden_file: The label string of the golden file containing the documentation when skydoc
          is run on the input file.
      skydoc: The label string of the skydoc binary.
      deps: A list of label strings of skylark file dependencies of the input_file.
    """
    output_golden_file = "%s_output.txt" % name
    native.sh_test(
        name = "%s_e2e_test" % name,
        srcs = ["skydoc_e2e_test_runner.sh"],
        args = [
            "$(location %s)" % skydoc,
            "$(location %s)" % input_file,
            "$(location %s)" % golden_file,
        ],
        data = [
            input_file,
            golden_file,
            skydoc,
        ] + deps,
    )

    native.genrule(
        name = "regenerate_%s_golden" % name,
        srcs = [
            input_file,
        ] + deps,
        outs = [output_golden_file],
        cmd = "$(location %s) " % skydoc +
              "$(location %s) $(location %s)" % (input_file, output_golden_file),
        tools = [skydoc],
    )
