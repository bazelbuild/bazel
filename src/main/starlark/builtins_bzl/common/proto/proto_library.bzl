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
load(":common/proto/proto_common.bzl", "get_import_path", proto_common = "proto_common_do_not_use")
load(":common/proto/proto_info.bzl", "ProtoInfo")
load(":common/paths.bzl", "paths")

def _check_srcs_package(target_package, srcs):
    """Check that .proto files in sources are from the same package.

    This is done to avoid clashes with the generated sources."""

    #TODO(bazel-team): this does not work with filegroups that contain files that are not in the package
    for src in srcs:
        if target_package != src.label.package:
            fail("Proto source with label '%s' must be in same package as consuming rule." % src.label)

def _get_import_prefix(ctx):
    """Gets and verifies import_prefix attribute if it is declared."""

    import_prefix = ctx.attr.import_prefix if hasattr(ctx.attr, "import_prefix") else ""

    if not paths.is_normalized(import_prefix):
        fail("should be normalized (without uplevel references or '.' path segments)", attr = "import_prefix")
    if paths.is_absolute(import_prefix):
        fail("should be a relative path", attr = "import_prefix")

    return import_prefix

def _get_strip_import_prefix(ctx):
    """Gets and verifies strip_import_prefix."""

    strip_import_prefix = ctx.attr.strip_import_prefix

    if not paths.is_normalized(strip_import_prefix):
        fail("should be normalized (without uplevel references or '.' path segments)", attr = "strip_import_prefix")

    if paths.is_absolute(strip_import_prefix):
        strip_import_prefix = strip_import_prefix[1:]
    else:  # Relative to current package
        strip_import_prefix = _join(ctx.label.package, strip_import_prefix)

    return strip_import_prefix.removesuffix("/")

def _proto_library_impl(ctx):
    semantics.preprocess(ctx)

    # Verifies attributes.
    _check_srcs_package(ctx.label.package, ctx.attr.srcs)
    srcs = ctx.files.srcs
    deps = [dep[ProtoInfo] for dep in ctx.attr.deps]
    exports = [dep[ProtoInfo] for dep in ctx.attr.exports]
    import_prefix = _get_import_prefix(ctx)
    strip_import_prefix = _get_strip_import_prefix(ctx)
    check_for_reexport = deps + exports if not srcs else exports
    for proto in check_for_reexport:
        if hasattr(proto, "allow_exports") and not proto.allow_exports.isAvailableFor(ctx.label):
            fail("proto_library '%s' can't be reexported in package '//%s'" % (proto.direct_descriptor_set.owner, ctx.label.package))

    proto_path, virtual_srcs = _process_srcs(ctx, srcs, import_prefix, strip_import_prefix)
    descriptor_set = ctx.actions.declare_file(ctx.label.name + "-descriptor-set.proto.bin")
    proto_info = ProtoInfo(
        srcs = virtual_srcs,
        deps = deps,
        descriptor_set = descriptor_set,
        proto_path = proto_path,
        workspace_root = ctx.label.workspace_root,
        bin_dir = ctx.bin_dir.path,
        allow_exports = ctx.attr.allow_exports,
    )

    _write_descriptor_set(ctx, proto_info, deps, exports, descriptor_set)

    # We assume that the proto sources will not have conflicting artifacts
    # with the same root relative path
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

def _process_srcs(ctx, srcs, import_prefix, strip_import_prefix):
    """Returns proto_path and sources, optionally symlinking them to _virtual_imports.

    Returns:
      (str, [File]) A pair of proto_path and virtual_sources.
    """
    if import_prefix != "" or strip_import_prefix != "":
        # Use virtual source roots
        return _symlink_to_virtual_imports(ctx, srcs, import_prefix, strip_import_prefix)
    else:
        # No virtual source roots
        return "", srcs

def _join(*path):
    return "/".join([p for p in path if p != ""])

def _symlink_to_virtual_imports(ctx, srcs, import_prefix, strip_import_prefix):
    """Symlinks srcs to _virtual_imports.

    Returns:
          A pair proto_path, directs_sources.
    """
    virtual_imports = _join("_virtual_imports", ctx.label.name)
    proto_path = _join(ctx.label.package, virtual_imports)

    if ctx.label.workspace_name == "":
        full_strip_import_prefix = strip_import_prefix
    else:
        full_strip_import_prefix = _join("..", ctx.label.workspace_name, strip_import_prefix)
    if full_strip_import_prefix:
        full_strip_import_prefix += "/"

    virtual_srcs = []
    for src in srcs:
        # Remove strip_import_prefix
        if not src.short_path.startswith(full_strip_import_prefix):
            fail(".proto file '%s' is not under the specified strip prefix '%s'" %
                 (src.short_path, full_strip_import_prefix))
        import_path = src.short_path[len(full_strip_import_prefix):]

        # Add import_prefix
        virtual_src = ctx.actions.declare_file(_join(virtual_imports, import_prefix, import_path))
        ctx.actions.symlink(
            output = virtual_src,
            target_file = src,
            progress_message = "Symlinking virtual .proto sources for %{label}",
        )
        virtual_srcs.append(virtual_src)
    return proto_path, virtual_srcs

def _write_descriptor_set(ctx, proto_info, deps, exports, descriptor_set):
    """Writes descriptor set."""
    if proto_info.direct_sources == []:
        ctx.actions.write(descriptor_set, "")
        return

    dependencies_descriptor_sets = depset(transitive = [dep.transitive_descriptor_sets for dep in deps])

    args = ctx.actions.args()
    if ctx.fragments.proto.experimental_proto_descriptorsets_include_source_info():
        args.add("--include_source_info")
    if hasattr(ctx.attr, "_retain_options") and ctx.attr._retain_options:
        args.add("--retain_options")

    strict_deps_mode = ctx.fragments.proto.strict_proto_deps()
    strict_deps = strict_deps_mode != "OFF" and strict_deps_mode != "DEFAULT"
    if strict_deps:
        if proto_info.direct_sources:
            strict_importable_sources = depset(
                direct = proto_info._direct_proto_sources,
                transitive = [dep._exported_sources for dep in deps],
            )
        else:
            strict_importable_sources = None
        if strict_importable_sources:
            args.add_joined("--direct_dependencies", strict_importable_sources, map_each = get_import_path, join_with = ":")
            # Example: `--direct_dependencies a.proto:b.proto`

        else:
            # The proto compiler requires an empty list to turn on strict deps checking
            args.add("--direct_dependencies=")

        # Set `-direct_dependencies_violation_msg=`
        args.add(ctx.label, format = semantics.STRICT_DEPS_FLAG_TEMPLATE)

    strict_public_imports_mode = ctx.fragments.proto.strict_public_imports()
    strict_imports = strict_public_imports_mode != "OFF" and strict_public_imports_mode != "DEFAULT"
    if strict_imports:
        public_import_protos = depset(transitive = [export._exported_sources for export in exports])
        if not public_import_protos:
            # This line is necessary to trigger the check.
            args.add("--allowed_public_imports=")
        else:
            args.add_joined("--allowed_public_imports", public_import_protos, map_each = get_import_path, join_with = ":")
    proto_lang_toolchain_info = proto_common.ProtoLangToolchainInfo(
        out_replacement_format_flag = "--descriptor_set_out=%s",
        output_files = "single",
        mnemonic = "GenProtoDescriptorSet",
        progress_message = "Generating Descriptor Set proto_library %{label}",
        proto_compiler = ctx.executable._proto_compiler,
        protoc_opts = ctx.fragments.proto.experimental_protoc_opts,
        plugin = None,
    )
    proto_common.compile(
        ctx.actions,
        proto_info,
        proto_lang_toolchain_info,
        generated_files = [descriptor_set],
        additional_inputs = dependencies_descriptor_sets,
        additional_args = args,
    )

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
        "strip_import_prefix": attr.string(default = "/"),
        "allow_exports": attr.label(
            cfg = "exec",
            providers = ["PackageSpecificationProvider"],
        ),
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
    exec_groups = semantics.EXEC_GROUPS,
)
