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
Definition of proto_common module.
"""

load(":common/proto/proto_semantics.bzl", "semantics")

ProtoInfo = _builtins.toplevel.ProtoInfo

def _write_descriptor_set(ctx, proto_info):
    output = proto_info.direct_descriptor_set
    proto_deps = [dep[ProtoInfo] for dep in ctx.attr.deps]
    dependencies_descriptor_sets = depset(transitive = [dep.transitive_descriptor_sets for dep in proto_deps])

    if proto_info.direct_sources == []:
        ctx.actions.write(output, "")
        return

    args = ctx.actions.args()
    args.use_param_file(param_file_arg = "@%s")
    args.set_param_file_format("multiline")
    args.add_all(proto_info.transitive_proto_path, map_each = _EXPAND_TRANSITIVE_PROTO_PATH_FLAGS)
    args.add(output, format = "--descriptor_set_out=%s")

    if ctx.fragments.proto.experimental_proto_descriptorsets_include_source_info():
        args.add("--include_source_info")
    args.add_all(ctx.fragments.proto.protoc_opts())

    strict_deps_mode = ctx.fragments.proto.strict_proto_deps()
    are_deps_strict = strict_deps_mode != "OFF" and strict_deps_mode != "DEFAULT"

    # Include maps

    # For each import, include both the import as well as the import relativized against its
    # protoSourceRoot. This ensures that protos can reference either the full path or the short
    # path when including other protos.
    args.add_all(proto_info.transitive_proto_sources(), map_each = _ExpandImportArgsFn)
    if are_deps_strict:
        strict_importable_sources = proto_info.strict_importable_sources()
        if strict_importable_sources:
            args.add_joined("--direct_dependencies", strict_importable_sources, map_each = _EXPAND_TO_IMPORT_PATHS, join_with = ":")
        else:
            # The proto compiler requires an empty list to turn on strict deps checking
            args.add("--direct_dependencies=")

        args.add(ctx.label, format = semantics.STRICT_DEPS_FLAG_TEMPLATE)

    # use exports

    args.add_all(proto_info.direct_sources)

    ctx.actions.run(
        mnemonic = "GenProtoDescriptorSet",
        progress_message = "Generating Descriptor Set proto_library %{label}",
        executable = ctx.executable._proto_compiler,
        arguments = [args],
        inputs = depset(transitive = [dependencies_descriptor_sets, proto_info.transitive_sources]),
        outputs = [output],
    )

def _EXPAND_TRANSITIVE_PROTO_PATH_FLAGS(flag):
    if flag == ".":
        return None
    return "--proto_path=" + flag

def _EXPAND_TO_IMPORT_PATHS(proto_source):
    return proto_source.import_path()

def _ExpandImportArgsFn(proto_source):
    return "-I%s=%s" % (proto_source.import_path(), proto_source.source_file().path)

proto_common = struct(
    write_descriptor_set = _write_descriptor_set,
)
