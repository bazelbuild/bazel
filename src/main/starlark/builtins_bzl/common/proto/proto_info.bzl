# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""
Definition of ProtoInfo provider.
"""

_warning = """ Don't use this field. It's intended for internal use and will be changed or removed
    without warning."""

ProtoInfo = provider(
    doc = "Encapsulates information provided by a `proto_library.`",
    fields = {
        "direct_sources": "(list[File]) The `.proto` source files from the `srcs` attribute.",
        "transitive_sources": """(depset[File]) The `.proto` source files from this rule and all
                    its dependent protocol buffer rules.""",
        "direct_descriptor_set": """(File) The descriptor set of the direct sources. If no srcs,
            contains an empty file.""",
        "transitive_descriptor_sets": """(depset[File]) A set of descriptor set files of all
            dependent `proto_library` rules, and this one's. This is not the same as passing
            --include_imports to proto-compiler. Will be empty if no dependencies.""",
        "proto_source_root": """(str) The directory relative to which the `.proto` files defined in
            the `proto_library` are defined. For example, if this is `a/b` and the rule has the
            file `a/b/c/d.proto` as a source, that source file would be imported as
            `import c/d.proto`

            In principle, the `proto_source_root` directory itself should always
            be relative to the output directory (`ctx.bin_dir` or `ctx.genfiles_dir`).

            This is at the moment not true for `proto_libraries` using (additional and/or strip)
            import prefixes. `proto_source_root` is in this case prefixed with the output
            directory. For example, the value is similar to
            `bazel-out/k8-fastbuild/bin/a/_virtual_includes/b` for an input file in
            `a/_virtual_includes/b/c.proto` that should be imported as `c.proto`.

            When using the value please account for both cases in a general way.
            That is assume the value is either prefixed with the output directory or not.
            This will make it possible to fix `proto_library` in the future.
            """,
        "transitive_proto_path": """(depset(str) A set of `proto_source_root`s collected from the
            transitive closure of this rule.""",
        "check_deps_sources": """(depset[File]) The `.proto` sources from the 'srcs' attribute.
            If the library is a proxy library that has no sources, it contains the
            `check_deps_sources` from this library's direct deps.""",

        # Deprecated fields:
        "transitive_imports": """(depset[File]) Deprecated: use `transitive_sources` instead.""",

        # Internal fields:
        "_direct_proto_sources": """(list[ProtoSourceInfo]) The `ProtoSourceInfo`s from the `srcs`
            attribute.""" + _warning,
        "_transitive_proto_sources": """(depset[ProtoSourceInfo]) The `ProtoSourceInfo`s from this
            rule and all its dependent protocol buffer rules.""" + _warning,
        "_exported_sources": """(depset[ProtoSourceInfo]) A set of `ProtoSourceInfo`s that may be
            imported by another `proto_library` depending on this one.""" + _warning,
    },
)

ProtoSourceInfo = provider(
    doc = "Represents a single `.proto` source file.",
    fields = {
        "_source_file": """(File) The `.proto` file. Possibly virtual to handle additional/stripped
          path prefix.""" + _warning,
        "_proto_path": "(str) The root of the virtual location." + _warning,
    },
)
