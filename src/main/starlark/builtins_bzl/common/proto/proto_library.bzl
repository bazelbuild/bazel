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
load(":common/paths.bzl", "paths")

ProtoInfo = _builtins.toplevel.ProtoInfo
native_proto_common = _builtins.toplevel.proto_common

def _check_srcs_package(target_package, srcs):
    """Makes sure the given srcs live in the given package."""
    for src in srcs:
        if target_package != src.label.package:
            fail("Proto source with label '%s' must be in same package as consuming rule." % src.label)

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
    descriptor_set = proto_info.direct_descriptor_set

    if proto_info.direct_sources == []:
        ctx.actions.write(descriptor_set, "")
        return

    deps = [dep[ProtoInfo] for dep in ctx.attr.deps]
    dependencies_descriptor_sets = depset(transitive = [dep.transitive_descriptor_sets for dep in deps])

    args = []
    if ctx.fragments.proto.experimental_proto_descriptorsets_include_source_info():
        args.append("--include_source_info")
    args.append((descriptor_set, "--descriptor_set_out=%s"))

    proto_common.create_proto_compile_action(
        ctx,
        proto_info,
        proto_compiler = ctx.executable._proto_compiler,
        mnemonic = "GenProtoDescriptorSet",
        progress_message = "Generating Descriptor Set proto_library %{label}",
        outputs = [descriptor_set],
        additional_inputs = dependencies_descriptor_sets,
        additional_args = args,
    )

def _proto_library_impl(ctx):
    semantics.preprocess(ctx)

    _check_srcs_package(ctx.label.package, ctx.attr.srcs)

    proto_info = _create_proto_info(ctx)

    _write_descriptor_set(ctx, proto_info)

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
        "strip_import_prefix": attr.string(default = "DO_NOT_STRIP"),
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
