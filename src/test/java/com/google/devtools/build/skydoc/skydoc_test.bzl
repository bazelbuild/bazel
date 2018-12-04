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

load("@bazel_skylib//:skylark_library.bzl", "bzl_library")
load("@skydoc//stardoc:stardoc.bzl", "stardoc")

def skydoc_test(name, input_file, golden_file, skydoc, deps = [], whitelisted_symbols = []):
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
      whitelisted_symbols: A list of strings representing top-level symbols in the input file
          to generate documentation for. If empty, documentation for all top-level symbols
          will be generated.
    """
    actual_generated_doc = "%s_output.txt" % name

    # Skydoc requires an absolute input file label to both load the target file and
    # track what its target is for the purpose of resolving relative labels.
    abs_input_file_label = str(Label("//%s" % native.package_name()).relative(input_file))

    native.sh_test(
        name = "%s_e2e_test" % name,
        srcs = ["diff_test_runner.sh"],
        args = [
            "$(location %s)" % actual_generated_doc,
            "$(location %s)" % golden_file,
        ],
        data = [
            actual_generated_doc,
            golden_file,
        ],
    )

    bzl_library(
        name = "%s_lib" % name,
        srcs = [input_file],
        deps = deps,
    )

    stardoc(
        name = "regenerate_%s_golden" % name,
        out = actual_generated_doc,
        input = input_file,
        symbol_names = whitelisted_symbols,
        deps = ["%s_lib" % name],
        stardoc = skydoc,
    )
