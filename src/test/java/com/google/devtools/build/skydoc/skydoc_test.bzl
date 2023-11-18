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

load("@rules_java//java:defs.bzl", "java_binary")

def _extract_binaryproto_impl(ctx):
    output_file = ctx.outputs.output
    extractor_args = ctx.actions.args()
    extractor_args.add("--input=" + str(ctx.file.input.owner))
    extractor_args.add("--output=" + output_file.path)
    extractor_args.add("--workspace_name=" + ctx.workspace_name)
    extractor_args.add_all(
        ctx.attr.symbol_names,
        format_each = "--symbols=%s",
        omit_if_empty = True,
    )
    ctx.actions.run(
        outputs = [output_file],
        executable = ctx.executable.tool,
        arguments = [extractor_args],
        mnemonic = "Stardoc",
        progress_message = ("Extracting Starlark doc for %s" % (ctx.label.name)),
    )
    outputs = [output_file]
    return [DefaultInfo(files = depset(outputs), runfiles = ctx.runfiles(files = outputs))]

extract_binaryproto = rule(
    doc = "Minimalistic binary-proto-only variant of the Stardoc rule using the legacy extractor",
    implementation = _extract_binaryproto_impl,
    attrs = {
        "input": attr.label(
            doc = "The starlark file to generate documentation for.",
            allow_single_file = [".bzl"],
            mandatory = True,
        ),
        "output": attr.output(
            doc = "The binary proto file to which documentation will be output.",
            mandatory = True,
        ),
        "symbol_names": attr.string_list(
            doc = """
A list of symbol names to generate documentation for. These should correspond to
the names of rule definitions in the input file. If this list is empty, then
documentation for all exported rule definitions will be generated.
""",
            default = [],
        ),
        "tool": attr.label(
            doc = "The location of the Stardoc legacy extractor tool.",
            allow_files = True,
            cfg = "exec",
            executable = True,
            mandatory = True,
        ),
    },
)

def skydoc_test(
        name,
        input_file,
        golden_file,
        deps = [],
        **kwargs):
    """Creates a test target and golden-file regeneration target for skydoc testing.

    The test target is named "{name}".
    The golden-file regeneration target is named "regenerate_{name}_golden".

    Args:
      name: A unique name to qualify the created targets.
      input_file: The label string of the Starlark input file for which documentation is generated
          in this test.
      golden_file: The label string of the golden file containing the documentation when skydoc
          is run on the input file.
      deps: A list of label strings of Starlark file dependencies of the input_file.
      **kwargs: Remaining arguments to passthrough to the underlying stardoc rule.
      """

    extractor = "%s_legacy_extractor" % name
    extractor_binary = "%s_binary" % extractor
    generated_binaryproto = "%s.binaryproto" % extractor
    generated_textproto = "%s.textproto" % extractor
    textproto_to_binaryproto = Label("//src/test/java/com/google/devtools/build/skydoc:binaryprotoToTextproto")

    native.sh_test(
        name = name,
        srcs = ["diff_test_runner.sh"],
        args = [
            "$(location %s)" % generated_textproto,
            "$(location %s)" % golden_file,
        ],
        data = [
            generated_textproto,
            golden_file,
        ],
    )

    java_binary(
        name = extractor_binary,
        main_class = "com.google.devtools.build.skydoc.SkydocMain",
        runtime_deps = [Label("//src/main/java/com/google/devtools/build/skydoc:skydoc_deploy.jar")],
        data = [input_file] + deps,
        tags = ["manual", "notap"],
        testonly = True,
        visibility = ["//visibility:private"],
    )

    extract_binaryproto(
        name = extractor,
        input = input_file,
        output = generated_binaryproto,
        tool = extractor_binary,
        testonly = True,
    )

    native.genrule(
        name = "regenerate_%s_golden" % name,
        srcs = [generated_binaryproto],
        outs = [generated_textproto],
        cmd = "./$(location %s) < $(location %s) > $(location %s)" % (
            textproto_to_binaryproto,
            generated_binaryproto,
            generated_textproto,
        ),
        tools = [textproto_to_binaryproto],
        testonly = True,
    )
