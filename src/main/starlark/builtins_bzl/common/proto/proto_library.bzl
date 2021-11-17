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

"""
Definition of proto_library rule.
"""

load(":common/proto/proto_semantics.bzl", "semantics")
load(":common/proto/proto_common.bzl", "proto_common")

ProtoInfo = _builtins.toplevel.ProtoInfo
native_proto_common = _builtins.toplevel.proto_common

def _check_srcs_package(target_package, srcs):
    """Makes sure the given srcs live in the given package."""
    for src in srcs:
        if target_package != src.label.package:
            fail("Proto source with label '%s' must be in same package as consuming rule." % src.label)

def _proto_library_impl(ctx):
    semantics.preprocess(ctx)

    _check_srcs_package(ctx.label.package, ctx.attr.srcs)

    proto_info = native_proto_common.create_proto_info(ctx)

    proto_common.write_descriptor_set(ctx, proto_info)

    data_runfiles = ctx.runfiles(
        files = [proto_info.direct_descriptor_set],
        transitive_files = depset(transitive = [proto_info.transitive_sources]),
    )

    return [
        proto_info,
        DefaultInfo(
            files = depset([proto_info.direct_descriptor_set]),
            default_runfiles = ctx.runfiles(),  # empty
            data_runfiles = data_runfiles,
        ),
    ]

proto_library = rule(
    _proto_library_impl,
    attrs = dict({
        "srcs": attr.label_list(
            allow_files = [".proto", ".protodevel"],
            flags = ["DIRECT_COMPILE_TIME_INPUT"],
        ),
        "deps": attr.label_list(
            providers = [ProtoInfo],
        ),
        "exports": attr.label_list(
            providers = [ProtoInfo],
        ),
        "strip_import_prefix": attr.string(),
        "data": attr.label_list(
            allow_files = True,
            flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
        ),
        "licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
        "_proto_compiler": attr.label(
            cfg = "exec",
            executable = True,
            allow_files = True,
            default = configuration_field("proto", "proto_compiler"),
        ),
    }, **semantics.EXTRA_ATTRIBUTES),
    fragments = ["proto"] + semantics.EXTRA_FRAGMENTS,
    provides = [ProtoInfo],
    output_to_genfiles = True,  # TODO(b/204266604) move to bin dir
)
