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
load(":common/proto/proto_common.bzl", proto_common = "proto_common_do_not_use")
load(":common/proto/proto_info.bzl", "ProtoInfo", "ProtoSourceInfo")
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

    return strip_import_prefix

def _proto_library_impl(ctx):
    semantics.preprocess(ctx)

    # Verifies attributes.
    _check_srcs_package(ctx.label.package, ctx.attr.srcs)
    srcs = ctx.files.srcs
    deps = [dep[ProtoInfo] for dep in ctx.attr.deps]
    exports = [dep[ProtoInfo] for dep in ctx.attr.exports]
    import_prefix = _get_import_prefix(ctx)
    strip_import_prefix = _get_strip_import_prefix(ctx)

    proto_path, direct_sources = _create_proto_sources(ctx, srcs, import_prefix, strip_import_prefix)
    descriptor_set = ctx.actions.declare_file(ctx.label.name + "-descriptor-set.proto.bin")
    proto_info = _create_proto_info(ctx, direct_sources, deps, exports, proto_path, descriptor_set)
    _write_descriptor_set(ctx, direct_sources, deps, exports, proto_info, descriptor_set)

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

def _create_proto_sources(ctx, srcs, import_prefix, strip_import_prefix):
    """Transforms Files in srcs to ProtoSourceInfos, optionally symlinking them to _virtual_imports.

    Returns:
      A pair proto_path, directs_sources.
    """
    generate_protos_in_virtual_imports = False
    if ctx.fragments.proto.generated_protos_in_virtual_imports():
        generate_protos_in_virtual_imports = any([not src.is_source for src in srcs])

    if import_prefix != "" or strip_import_prefix != "" or generate_protos_in_virtual_imports:
        # Use virtual source roots
        return _symlink_to_virtual_imports(ctx, srcs, import_prefix, strip_import_prefix)
    else:
        # No virtual source roots
        direct_sources = []
        for src in srcs:
            if ctx.label.workspace_name == "" or ctx.label.workspace_root.startswith(".."):
                # source_root == ''|'bazel-out/foo/k8-fastbuild/bin'
                source_root = src.root.path
            else:
                # source_root == ''|'bazel-out/foo/k8-fastbuild/bin' / 'external/repo'
                source_root = _join(src.root.path, ctx.label.workspace_root)
            direct_sources.append(ProtoSourceInfo(_source_file = src, _original_source_file = src, _proto_path = source_root))

        return ctx.label.workspace_root if ctx.label.workspace_root else ".", direct_sources

def _join(*path):
    return "/".join([p for p in path if p != ""])

def _symlink_to_virtual_imports(ctx, srcs, import_prefix, strip_import_prefix):
    """Symlinks srcs to _virtual_imports.

    Returns:
          A pair proto_path, directs_sources.
    """
    virtual_imports = _join("_virtual_imports", ctx.label.name)
    if ctx.label.workspace_name == "" or ctx.label.workspace_root.startswith(".."):  # siblingRepositoryLayout
        # Example: `bazel-out/[repo/]target/bin / pkg / _virtual_imports/name`
        proto_path = _join(ctx.genfiles_dir.path, ctx.label.package, virtual_imports)
    else:
        # Example: `bazel-out/target/bin / repo / pkg / _virtual_imports/name`
        proto_path = _join(ctx.genfiles_dir.path, ctx.label.workspace_root, ctx.label.package, virtual_imports)

    direct_sources = []
    for src in srcs:
        if ctx.label.workspace_name == "":
            repository_relative_path = src.short_path
        else:
            # src.short_path = ../repo/pkg/a.proto
            repository_relative_path = paths.relativize(src.short_path, "../" + ctx.label.workspace_name)

        # Remove strip_import_prefix
        if not repository_relative_path.startswith(strip_import_prefix):
            fail(".proto file '%s' is not under the specified strip prefix '%s'" %
                 (src.short_path, strip_import_prefix))
        import_path = repository_relative_path[len(strip_import_prefix):]

        # Add import_prefix
        virtual_src = ctx.actions.declare_file(_join(virtual_imports, import_prefix, import_path))

        ctx.actions.symlink(
            output = virtual_src,
            target_file = src,
            progress_message = "Symlinking virtual .proto sources for %{label}",
        )
        direct_sources.append(ProtoSourceInfo(_source_file = virtual_src, _original_source_file = src, _proto_path = proto_path))
    return proto_path, direct_sources

def _create_proto_info(ctx, direct_sources, deps, exports, proto_path, descriptor_set):
    """Constructs ProtoInfo."""

    # Construct ProtoInfo
    transitive_proto_sources = depset(
        direct = direct_sources,
        transitive = [dep._transitive_proto_sources for dep in deps],
        order = "preorder",
    )
    transitive_sources = depset(
        direct = [src._source_file for src in direct_sources],
        transitive = [dep.transitive_sources for dep in deps],
        order = "preorder",
    )
    transitive_proto_path = depset(
        direct = [proto_path],
        transitive = [dep.transitive_proto_path for dep in deps],
    )
    if direct_sources:
        check_deps_sources = depset(direct = [src._source_file for src in direct_sources])
    else:
        check_deps_sources = depset(transitive = [dep.check_deps_sources for dep in deps])

    transitive_descriptor_sets = depset(
        direct = [descriptor_set],
        transitive = [dep.transitive_descriptor_sets for dep in deps],
    )

    # Layering checks.
    if direct_sources:
        exported_sources = depset(direct = direct_sources)
    else:
        exported_sources = depset(transitive = [dep._exported_sources for dep in deps])

    return ProtoInfo(
        direct_sources = [src._source_file for src in direct_sources],
        transitive_sources = transitive_sources,
        direct_descriptor_set = descriptor_set,
        transitive_descriptor_sets = transitive_descriptor_sets,
        proto_source_root = proto_path,
        transitive_proto_path = transitive_proto_path,
        check_deps_sources = check_deps_sources,
        transitive_imports = transitive_sources,
        _direct_proto_sources = direct_sources,
        _transitive_proto_sources = transitive_proto_sources,
        _exported_sources = exported_sources,
    )

def _get_import_path(proto_source):
    return paths.relativize(proto_source._source_file.path, proto_source._proto_path)

def _write_descriptor_set(ctx, direct_sources, deps, exports, proto_info, descriptor_set):
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
        if direct_sources:
            strict_importable_sources = depset(
                direct = direct_sources,
                transitive = [dep._exported_sources for dep in deps],
            )
        else:
            strict_importable_sources = None
        if strict_importable_sources:
            args.add_joined("--direct_dependencies", strict_importable_sources, map_each = _get_import_path, join_with = ":")
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
            args.add_joined("--allowed_public_imports", public_import_protos, map_each = _get_import_path, join_with = ":")
    proto_lang_toolchain_info = proto_common.ProtoLangToolchainInfo(
        out_replacement_format_flag = "--descriptor_set_out=%s",
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
        plugin_output = descriptor_set,
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
    exec_groups = semantics.EXEC_GROUPS,
)
