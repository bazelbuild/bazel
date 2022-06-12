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
load(":common/paths.bzl", "paths")

ProtoInfo = _builtins.toplevel.ProtoInfo
native_proto_common = _builtins.toplevel.proto_common

def _join(*path):
    return "/".join([p for p in path if p != ""])

def _create_proto_info(ctx):
    srcs = ctx.files.srcs
    deps = [dep[ProtoInfo] for dep in ctx.attr.deps]
    exports = [dep[ProtoInfo] for dep in ctx.attr.exports]

    import_prefix = ctx.attr.import_prefix if hasattr(ctx.attr, "import_prefix") else ""
    if not paths.is_normalized(import_prefix):
        fail("should be normalized (without uplevel references or '.' path segments)", attr = "import_prefix")

    strip_import_prefix = ctx.attr.strip_import_prefix
    if not paths.is_normalized(strip_import_prefix):
        fail("should be normalized (without uplevel references or '.' path segments)", attr = "strip_import_prefix")
    if strip_import_prefix.startswith("/"):
        strip_import_prefix = strip_import_prefix[1:]
    elif strip_import_prefix != "DO_NOT_STRIP":  # Relative to current package
        strip_import_prefix = _join(ctx.label.package, strip_import_prefix)
    else:
        strip_import_prefix = ""

    has_generated_sources = False
    if ctx.fragments.proto.generated_protos_in_virtual_imports():
        has_generated_sources = any([not src.is_source for src in srcs])

    direct_sources = []
    if import_prefix != "" or strip_import_prefix != "" or has_generated_sources:
        # Use virtual source roots
        if paths.is_absolute(import_prefix):
            fail("should be a relative path", attr = "import_prefix")

        virtual_imports = _join("_virtual_imports", ctx.label.name)
        if ctx.label.workspace_name == "" or ctx.label.workspace_root.startswith(".."):  # siblingRepositoryLayout
            proto_path = _join(ctx.genfiles_dir.path, ctx.label.package, virtual_imports)
        else:
            proto_path = _join(ctx.genfiles_dir.path, ctx.label.workspace_root, ctx.label.package, virtual_imports)

        for src in srcs:
            if ctx.label.workspace_name == "":
                repository_relative_path = src.short_path
            else:
                repository_relative_path = paths.relativize(src.short_path, "../" + ctx.label.workspace_name)

            if not repository_relative_path.startswith(strip_import_prefix):
                fail(".proto file '%s' is not under the specified strip prefix '%s'" %
                     (src.short_path, strip_import_prefix))
            import_path = repository_relative_path[len(strip_import_prefix):]

            virtual_src = ctx.actions.declare_file(_join(virtual_imports, import_prefix, import_path))
            ctx.actions.symlink(
                output = virtual_src,
                target_file = src,
                progress_message = "Symlinking virtual .proto sources for %{label}",
            )
            direct_sources.append(native_proto_common.ProtoSource(virtual_src, src, proto_path))

    else:
        # No virtual source roots
        proto_path = "."
        for src in srcs:
            direct_sources.append(native_proto_common.ProtoSource(src, src, ctx.label.workspace_root + src.root.path))

    # Construct ProtoInfo
    transitive_proto_sources = depset(
        direct = direct_sources,
        transitive = [dep.transitive_proto_sources() for dep in deps],
        order = "preorder",
    )
    transitive_sources = depset(
        direct = [src.source_file() for src in direct_sources],
        transitive = [dep.transitive_sources for dep in deps],
        order = "preorder",
    )
    transitive_proto_path = depset(
        direct = [proto_path],
        transitive = [dep.transitive_proto_path for dep in deps],
    )
    if direct_sources:
        check_deps_sources = depset(direct = [src.source_file() for src in direct_sources])
    else:
        check_deps_sources = depset(transitive = [dep.check_deps_sources for dep in deps])

    direct_descriptor_set = ctx.actions.declare_file(ctx.label.name + "-descriptor-set.proto.bin")
    transitive_descriptor_sets = depset(
        direct = [direct_descriptor_set],
        transitive = [dep.transitive_descriptor_sets for dep in deps],
    )

    # Layering checks.
    if direct_sources:
        exported_sources = depset(direct = direct_sources)
        strict_importable_sources = depset(
            direct = direct_sources,
            transitive = [dep.exported_sources() for dep in deps],
        )
    else:
        exported_sources = depset(transitive = [dep.exported_sources() for dep in deps])
        strict_importable_sources = depset()
    public_import_protos = depset(transitive = [export.exported_sources() for export in exports])

    return native_proto_common.ProtoInfo(
        direct_sources,
        proto_path,
        transitive_sources,
        transitive_proto_sources,
        transitive_proto_path,
        check_deps_sources,
        direct_descriptor_set,
        transitive_descriptor_sets,
        exported_sources,
        strict_importable_sources,
        public_import_protos,
    )

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
    create_proto_info = _create_proto_info,
    write_descriptor_set = _write_descriptor_set,
)
