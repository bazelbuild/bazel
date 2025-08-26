# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""Starlark implementation of cc_toolchain rule."""

load(":common/cc/cc_common.bzl", "cc_common")
load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/cc/cc_toolchain_provider_helper.bzl", "get_cc_toolchain_provider")
load(":common/cc/fdo/fdo_context.bzl", "create_fdo_context")
load(":common/cc/semantics.bzl", "semantics")

cc_internal = _builtins.internal.cc_internal
ToolchainInfo = _builtins.toplevel.platform_common.ToolchainInfo
TemplateVariableInfo = _builtins.toplevel.platform_common.TemplateVariableInfo
PackageSpecificationInfo = _builtins.toplevel.PackageSpecificationInfo
CcToolchainConfigInfo = _builtins.toplevel.CcToolchainConfigInfo

def _files(ctx, attr_name):
    attr = getattr(ctx.attr, attr_name, None)
    files = []
    if attr != None and DefaultInfo in attr:
        # NOTE: This does not currently handle symlinks.
        # Also note: this does not cause the .runfiles/ tree to be materialized.
        # TODO: Modify the link action to properly handle runfiles.
        files.append(attr[DefaultInfo].files)
        files.append(attr[DefaultInfo].default_runfiles.files)
        return depset(transitive = files)
    return depset()

def _provider(attr, provider):
    if attr != None and provider in attr:
        return attr[provider]
    return None

def _latebound_libc(ctx, attr_name, implicit_attr_name):
    if getattr(ctx.attr, implicit_attr_name, None) == None:
        return attr_name
    return implicit_attr_name

def _full_inputs_for_link(ctx, linker_files, libc):
    return depset(
        [ctx.file._interface_library_builder, ctx.file._link_dynamic_library_tool],
        transitive = [linker_files, libc],
    )

def _label(ctx, attr_name):
    if getattr(ctx.attr, attr_name, None) != None:
        return getattr(ctx.attr, attr_name).label
    return None

def _package_specification_provider(ctx, allowlist_name):
    possible_attr_names = ["_whitelist_" + allowlist_name, "_allowlist_" + allowlist_name]
    for attr_name in possible_attr_names:
        if hasattr(ctx.attr, attr_name):
            package_specification_provider = getattr(ctx.attr, attr_name)[PackageSpecificationInfo]
            if package_specification_provider != None:
                return package_specification_provider
    fail("Allowlist argument for " + allowlist_name + " not found")

def _single_file(ctx, attr_name):
    files = getattr(ctx.files, attr_name, [])
    if len(files) > 1:
        fail(ctx.label.name + " expected a single artifact", attr = attr_name)
    if len(files) == 1:
        return files[0]
    return None

def _attributes(ctx):
    grep_includes = None
    if not semantics.is_bazel:
        grep_includes = _single_file(ctx, "_grep_includes")

    latebound_libc = _latebound_libc(ctx, "libc_top", "_libc_top")
    latebound_target_libc = _latebound_libc(ctx, "libc_top", "_target_libc_top")

    all_files = _files(ctx, "all_files")
    return struct(
        supports_param_files = ctx.attr.supports_param_files,
        runtime_solib_dir_base = "_solib__" + cc_internal.escape_label(label = ctx.label),
        cc_toolchain_config_info = _provider(ctx.attr.toolchain_config, CcToolchainConfigInfo),
        static_runtime_lib = ctx.attr.static_runtime_lib,
        dynamic_runtime_lib = ctx.attr.dynamic_runtime_lib,
        supports_header_parsing = ctx.attr.supports_header_parsing,
        all_files = all_files,
        compiler_files = _files(ctx, "compiler_files"),
        strip_files = _files(ctx, "strip_files"),
        objcopy_files = _files(ctx, "objcopy_files"),
        link_dynamic_library_tool = ctx.file._link_dynamic_library_tool,
        grep_includes = grep_includes,
        aggregate_ddi = _single_file(ctx, "_aggregate_ddi"),
        generate_modmap = ctx.attr.generate_modmap[DefaultInfo].files_to_run if getattr(ctx.attr, "generate_modmap", None) else None,
        module_map = ctx.attr.module_map,
        as_files = _files(ctx, "as_files"),
        ar_files = _files(ctx, "ar_files"),
        dwp_files = _files(ctx, "dwp_files"),
        module_map_artifact = _single_file(ctx, "module_map"),
        all_files_including_libc = depset(transitive = [_files(ctx, "all_files"), _files(ctx, latebound_libc)]),
        zipper = ctx.file._zipper,
        linker_files = _full_inputs_for_link(
            ctx,
            _files(ctx, "linker_files"),
            _files(ctx, latebound_libc),
        ),
        cc_toolchain_label = ctx.label,
        coverage_files = _files(ctx, "coverage_files") or all_files,
        compiler_files_without_includes = _files(ctx, "compiler_files_without_includes"),
        libc = _files(ctx, latebound_libc),
        target_libc = _files(ctx, latebound_target_libc),
        libc_top_label = _label(ctx, latebound_libc),
        target_libc_top_label = _label(ctx, latebound_target_libc),
        if_so_builder = ctx.file._interface_library_builder,
        allowlist_for_layering_check = _package_specification_provider(ctx, "disabling_parse_headers_and_layering_check_allowed"),
        build_info_files = _provider(ctx.attr._build_info_translator, OutputGroupInfo),
    )

def _cc_toolchain_impl(ctx):
    attributes = _attributes(ctx)
    providers = []

    cc_toolchain = get_cc_toolchain_provider(ctx, attributes)
    if cc_toolchain == None:
        fail("This should never happen")
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    template_vars = cc_toolchain._additional_make_variables | cc_helper.get_toolchain_global_make_variables(cc_toolchain) | cc_helper.get_cc_flags_make_variable(ctx, feature_configuration, cc_toolchain)
    template_variable_info = TemplateVariableInfo(template_vars)
    toolchain = ToolchainInfo(
        cc = cc_toolchain,
        # Add a clear signal that this is a CcToolchainProvider, since just "cc" is
        # generic enough to possibly be re-used.
        cc_provider_in_toolchain = True,
    )
    providers.append(cc_toolchain)
    providers.append(toolchain)
    providers.append(template_variable_info)
    providers.append(DefaultInfo(files = cc_toolchain._all_files_including_libc))
    return providers

cc_toolchain = rule(
    implementation = _cc_toolchain_impl,
    fragments = ["cpp"],
    doc = """
<p>Represents a C++ toolchain.</p>

<p>
  This rule is responsible for:

  <ul>
    <li>
      Collecting all artifacts needed for C++ actions to run. This is done by
      attributes such as <code>all_files</code>, <code>compiler_files</code>,
      <code>linker_files</code>, or other attributes ending with <code>_files</code>). These are
      most commonly filegroups globbing all required files.
    </li>
    <li>
      Generating correct command lines for C++ actions. This is done using
      <code>CcToolchainConfigInfo</code> provider (details below).
    </li>
  </ul>
</p>
<p>
  Use <code>toolchain_config</code> attribute to configure the C++ toolchain.
  See also this
  <a href="https://bazel.build/docs/cc-toolchain-config-reference">
    page
  </a> for elaborate C++ toolchain configuration and toolchain selection documentation.
</p>
<p>
  Use <code>tags = ["manual"]</code> in order to prevent toolchains from being built and configured
  unnecessarily when invoking <code>bazel build //...</code>
</p>""",
    subrules = [create_fdo_context],
    attrs = {
        # buildifier: disable=attr-license
        "licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
        # buildifier: disable=attr-license
        "output_licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
        "toolchain_identifier": attr.string(
            default = "",
            doc = """
The identifier used to match this cc_toolchain with the corresponding
crosstool_config.toolchain.

<p>
  Until issue <a href="https://github.com/bazelbuild/bazel/issues/5380">#5380</a> is fixed
  this is the recommended way of associating <code>cc_toolchain</code> with
  <code>CROSSTOOL.toolchain</code>. It will be replaced by the <code>toolchain_config</code>
  attribute (<a href="https://github.com/bazelbuild/bazel/issues/5380">#5380</a>).</p>""",
        ),
        "all_files": attr.label(
            allow_files = True,
            mandatory = True,
            doc = """
Collection of all cc_toolchain artifacts. These artifacts will be added as inputs to all
rules_cc related actions (with the exception of actions that are using more precise sets of
artifacts from attributes below). Bazel assumes that <code>all_files</code> is a superset
of all other artifact-providing attributes (e.g. linkstamp compilation needs both compile
and link files, so it takes <code>all_files</code>).

<p>
This is what <code>cc_toolchain.files</code> contains, and this is used by all Starlark
rules using C++ toolchain.</p>""",
        ),
        "compiler_files": attr.label(
            allow_files = True,
            mandatory = True,
            doc = """
Collection of all cc_toolchain artifacts required for compile actions.""",
        ),
        "compiler_files_without_includes": attr.label(
            allow_files = True,
            doc = """
Collection of all cc_toolchain artifacts required for compile actions in case when
input discovery is supported (currently Google-only).""",
        ),
        "strip_files": attr.label(
            allow_files = True,
            mandatory = True,
            doc = """
Collection of all cc_toolchain artifacts required for strip actions.""",
        ),
        "objcopy_files": attr.label(
            allow_files = True,
            mandatory = True,
            doc = """
Collection of all cc_toolchain artifacts required for objcopy actions.""",
        ),
        "as_files": attr.label(
            allow_files = True,
            doc = """
Collection of all cc_toolchain artifacts required for assembly actions.""",
        ),
        "ar_files": attr.label(
            allow_files = True,
            doc = """
Collection of all cc_toolchain artifacts required for archiving actions.""",
        ),
        "linker_files": attr.label(
            allow_files = True,
            mandatory = True,
            doc = """
Collection of all cc_toolchain artifacts required for linking actions.""",
        ),
        "dwp_files": attr.label(
            allow_files = True,
            mandatory = True,
            doc = """
Collection of all cc_toolchain artifacts required for dwp actions.""",
        ),
        "coverage_files": attr.label(
            allow_files = True,
            doc = """
Collection of all cc_toolchain artifacts required for coverage actions. If not specified,
all_files are used.""",
        ),
        "libc_top": attr.label(
            # TODO(b/78578234): Make this the default and remove the late-bound versions.
            allow_files = False,
            doc = """
A collection of artifacts for libc passed as inputs to compile/linking actions.""",
        ),
        "static_runtime_lib": attr.label(
            allow_files = True,
            doc = """
Static library artifact for the C++ runtime library (e.g. libstdc++.a).

<p>This will be used when 'static_link_cpp_runtimes' feature is enabled, and we're linking
dependencies statically.</p>""",
        ),
        "dynamic_runtime_lib": attr.label(
            allow_files = True,
            doc = """
Dynamic library artifact for the C++ runtime library (e.g. libstdc++.so).

<p>This will be used when 'static_link_cpp_runtimes' feature is enabled, and we're linking
dependencies dynamically.</p>""",
        ),
        "module_map": attr.label(
            allow_files = True,
            doc = """
Module map artifact to be used for modular builds.""",
        ),
        "supports_param_files": attr.bool(
            default = True,
            doc = """
Set to True when cc_toolchain supports using param files for linking actions.""",
        ),
        "supports_header_parsing": attr.bool(
            default = False,
            doc = """
Set to True when cc_toolchain supports header parsing actions.""",
        ),
        "exec_transition_for_inputs": attr.bool(
            default = False,  # No-op.
            doc = "Deprecated. No-op.",
        ),
        "toolchain_config": attr.label(
            allow_files = False,
            mandatory = True,
            providers = [CcToolchainConfigInfo],
            doc = """
The label of the rule providing <code>cc_toolchain_config_info</code>.""",
        ),
        "_libc_top": attr.label(
            default = configuration_field(fragment = "cpp", name = "libc_top"),
        ),
        "_grep_includes": semantics.get_grep_includes(),
        "_interface_library_builder": attr.label(
            default = "@" + semantics.get_repo() + "//tools/cpp:interface_library_builder",
            allow_single_file = True,
            cfg = "exec",
        ),
        "_link_dynamic_library_tool": attr.label(
            default = "@" + semantics.get_repo() + "//tools/cpp:link_dynamic_library",
            allow_single_file = True,
            cfg = "exec",
        ),
        "_cc_toolchain_type": attr.label(default = "@" + semantics.get_repo() + "//tools/cpp:toolchain_type"),
        "_zipper": attr.label(
            default = configuration_field(fragment = "cpp", name = "zipper"),
            allow_single_file = True,
            cfg = "exec",
        ),
        "_target_libc_top": attr.label(
            default = configuration_field(fragment = "cpp", name = "target_libc_top_DO_NOT_USE_ONLY_FOR_CC_TOOLCHAIN"),
        ),
        "_whitelist_disabling_parse_headers_and_layering_check_allowed": attr.label(
            default = "@" + semantics.get_repo() + "//tools/build_defs/cc/whitelists/parse_headers_and_layering_check:disabling_parse_headers_and_layering_check_allowed",
            providers = [PackageSpecificationInfo],
        ),
        "_build_info_translator": attr.label(
            default = semantics.BUILD_INFO_TRANLATOR_LABEL,
            providers = [OutputGroupInfo],
        ),
    } | semantics.cpp_modules_tools(),
)
