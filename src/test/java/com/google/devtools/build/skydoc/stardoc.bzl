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

"""Starlark rule for stardoc: a documentation generator tool written in Java."""

load("//src/test/java/com/google/devtools/build/skydoc/private:stardoc.bzl", _stardoc = "stardoc")

def stardoc(
        *,
        name,
        input,
        out,
        deps = [],
        format = "markdown",
        symbol_names = [],
        semantic_flags = [],
        stardoc = Label("//src/main/java/com/google/devtools/build/skydoc:skydoc_deploy.jar"),
        renderer = Label("//src/main/java/com/google/devtools/build/skydoc/renderer"),
        aspect_template = Label(":test_templates/markdown_tables/aspect.vm"),
        func_template = Label(":test_templates/markdown_tables/func.vm"),
        header_template = Label(":test_templates/markdown_tables/header.vm"),
        provider_template = Label(":test_templates/markdown_tables/provider.vm"),
        rule_template = Label(":test_templates/markdown_tables/rule.vm"),
        **kwargs):
    """Generates documentation for exported starlark rule definitions in a target starlark file.

    Args:
      name: The name of the stardoc target.
      input: The starlark file to generate documentation for (mandatory).
      out: The file to which documentation will be output (mandatory).
      deps: A list of bzl_library dependencies which the input depends on.
      format: The format of the output file. Valid values: 'markdown' or 'proto'.
      symbol_names: A list of symbol names to generate documentation for. These should correspond to the names of rule
        definitions in the input file. If this list is empty, then documentation for all exported rule definitions will
        be generated.
      semantic_flags: A list of canonical flags to affect Starlark semantics for the Starlark interpreter during
        documentation generation. This should only be used to maintain compatibility with non-default semantic flags
        required to use the given Starlark symbols.

        For example, if `//foo:bar.bzl` does not build except when a user would specify
        `--incompatible_foo_semantic=false`, then this attribute should contain
        "--incompatible_foo_semantic=false".
      stardoc: The location of the stardoc tool.
      renderer: The location of the renderer tool.
      aspect_template: The input file template for generating documentation of aspects
      header_template: The input file template for the header of the output documentation.
      func_template: The input file template for generating documentation of functions.
      provider_template: The input file template for generating documentation of providers.
      rule_template: The input file template for generating documentation of rules.
      **kwargs: Further arguments to pass to stardoc.
    """

    stardoc_with_runfiles_name = name + "_stardoc"

    testonly = {"testonly": kwargs["testonly"]} if "testonly" in kwargs else {}
    native.java_binary(
        name = stardoc_with_runfiles_name,
        main_class = "com.google.devtools.build.skydoc.SkydocMain",
        runtime_deps = [stardoc],
        data = [input] + deps,
        tags = ["manual"],
        visibility = ["//visibility:private"],
        **testonly
    )

    _stardoc(
        name = name,
        input = input,
        out = out,
        format = format,
        symbol_names = symbol_names,
        semantic_flags = semantic_flags,
        stardoc = stardoc_with_runfiles_name,
        renderer = renderer,
        aspect_template = aspect_template,
        func_template = func_template,
        header_template = header_template,
        provider_template = provider_template,
        rule_template = rule_template,
        **kwargs
    )
