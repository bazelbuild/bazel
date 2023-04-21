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

load("@bazel_skylib//:bzl_library.bzl", "StarlarkLibraryInfo")

def _root_from_file(f):
    """Given a file, returns the root path of that file."""
    return f.root.path or "."

def _stardoc_impl(ctx):
    """Implementation of the stardoc rule."""
    for semantic_flag in ctx.attr.semantic_flags:
        if not semantic_flag.startswith("--"):
            fail("semantic_flags entry '%s' must start with '--'" % semantic_flag)
    out_file = ctx.outputs.out
    input_files = depset(direct = [ctx.file.input], transitive = [
        dep[StarlarkLibraryInfo].transitive_srcs
        for dep in ctx.attr.deps
    ])
    stardoc_args = ctx.actions.args()
    stardoc_args.add("--input=" + str(ctx.file.input.owner))
    stardoc_args.add("--workspace_name=" + ctx.workspace_name)
    stardoc_args.add_all(
        ctx.attr.symbol_names,
        format_each = "--symbols=%s",
        omit_if_empty = True,
    )

    # TODO(cparsons): Note that use of dep_roots alone does not guarantee
    # the correct file is loaded. If two files exist under the same path
    # but are under different roots, it is possible that Stardoc loads the
    # one that is not explicitly an input to this action (if sandboxing is
    # disabled). The correct way to resolve this is to explicitly specify
    # the full set of transitive dependency Starlark files as action args
    # (maybe using a param file), but this requires some work.
    stardoc_args.add_all(
        input_files,
        format_each = "--dep_roots=%s",
        map_each = _root_from_file,
        omit_if_empty = True,
        uniquify = True,
    )

    # Needed in case some files are referenced across local repository
    # namespace. For example, consider a file under a nested local repository @bar
    # rooted under ./foo/bar/WORKSPACE. Consider a stardoc target 'lib_doc' under
    # foo/bar/BUILD to document foo/bar/lib.bzl.
    # The stardoc target references @bar//third_party/stardoc:lib.bzl (which appears just as :lib.bzl), but the
    # actual build is taking place in the root repository, thus the source file
    # is present under external/bar/lib.bzl.
    stardoc_args.add(
        "--dep_roots=external/" + ctx.workspace_name,
    )
    stardoc_args.add_all(ctx.attr.semantic_flags)
    stardoc = ctx.executable.stardoc

    if ctx.attr.format == "proto":
        stardoc_args.add("--output=" + out_file.path)
        ctx.actions.run(
            outputs = [out_file],
            inputs = input_files,
            executable = stardoc,
            arguments = [stardoc_args],
            mnemonic = "Stardoc",
            progress_message = ("Generating Starlark doc for %s" %
                                (ctx.label.name)),
        )
    elif ctx.attr.format == "markdown":
        proto_file = ctx.actions.declare_file(ctx.label.name + ".raw", sibling = out_file)
        stardoc_args.add("--output=" + proto_file.path)
        ctx.actions.run(
            outputs = [proto_file],
            inputs = input_files,
            executable = stardoc,
            arguments = [stardoc_args],
            mnemonic = "Stardoc",
            progress_message = ("Generating proto for Starlark doc for %s" %
                                (ctx.label.name)),
        )
        renderer_args = ctx.actions.args()
        renderer_args.add("--input=" + str(proto_file.path))
        renderer_args.add("--output=" + str(ctx.outputs.out.path))
        renderer_args.add("--aspect_template=" + str(ctx.file.aspect_template.path))
        renderer_args.add("--header_template=" + str(ctx.file.header_template.path))
        renderer_args.add("--func_template=" + str(ctx.file.func_template.path))
        renderer_args.add("--provider_template=" + str(ctx.file.provider_template.path))
        renderer_args.add("--rule_template=" + str(ctx.file.rule_template.path))
        renderer = ctx.executable.renderer
        ctx.actions.run(
            outputs = [out_file],
            inputs = [proto_file, ctx.file.aspect_template, ctx.file.header_template, ctx.file.func_template, ctx.file.provider_template, ctx.file.rule_template],
            executable = renderer,
            arguments = [renderer_args],
            mnemonic = "Renderer",
            progress_message = ("Converting proto format of %s to markdown format" %
                                (ctx.label.name)),
        )

    # Work around default outputs not getting captured by sh_binary:
    # https://github.com/bazelbuild/bazel/issues/15043.
    # See discussion in https://github.com/bazelbuild/stardoc/pull/139.
    outputs = [out_file]
    return [DefaultInfo(files = depset(outputs), runfiles = ctx.runfiles(files = outputs))]

stardoc = rule(
    _stardoc_impl,
    doc = """
Generates documentation for exported skylark rule definitions in a target starlark file.

This rule is an experimental replacement for the existing skylark_doc rule.
""",
    attrs = {
        "input": attr.label(
            doc = "The starlark file to generate documentation for.",
            allow_single_file = [".bzl"],
        ),
        "deps": attr.label_list(
            doc = "A list of bzl_library dependencies which the input depends on.",
            providers = [StarlarkLibraryInfo],
        ),
        "format": attr.string(
            doc = "The format of the output file. Valid values: 'markdown' or 'proto'.",
            default = "markdown",
            values = ["markdown", "proto"],
        ),
        "out": attr.output(
            doc = "The (markdown) file to which documentation will be output.",
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
        "semantic_flags": attr.string_list(
            doc = """
A list of canonical flags to affect Starlark semantics for the Starlark interpretter
during documentation generation. This should only be used to maintain compatibility with
non-default semantic flags required to use the given Starlark symbols.

For example, if `//foo:bar.bzl` does not build except when a user would specify
`--incompatible_foo_semantic=false`, then this attribute should contain
"--incompatible_foo_semantic=false".
""",
            default = [],
        ),
        "stardoc": attr.label(
            doc = "The location of the stardoc tool.",
            allow_files = True,
            default = Label("//src/main/java/com/google/devtools/build/skydoc"),
            cfg = "exec",
            executable = True,
        ),
        "renderer": attr.label(
            doc = "The location of the renderer tool.",
            allow_files = True,
            default = Label("//src/main/java/com/google/devtools/build/skydoc/renderer"),
            cfg = "exec",
            executable = True,
        ),
        "aspect_template": attr.label(
            doc = "The input file template for generating documentation of aspects.",
            allow_single_file = [".vm"],
            default = Label(":test_templates/markdown_tables/aspect.vm"),
        ),
        "header_template": attr.label(
            doc = "The input file template for the header of the output documentation.",
            allow_single_file = [".vm"],
            default = Label(":test_templates/markdown_tables/header.vm"),
        ),
        "func_template": attr.label(
            doc = "The input file template for generating documentation of functions.",
            allow_single_file = [".vm"],
            default = Label(":test_templates/markdown_tables/func.vm"),
        ),
        "provider_template": attr.label(
            doc = "The input file template for generating documentation of providers.",
            allow_single_file = [".vm"],
            default = Label(":test_templates/markdown_tables/provider.vm"),
        ),
        "rule_template": attr.label(
            doc = "The input file template for generating documentation of rules.",
            allow_single_file = [".vm"],
            default = Label(":test_templates/markdown_tables/rule.vm"),
        ),
    },
)
