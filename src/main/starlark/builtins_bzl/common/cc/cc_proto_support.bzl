# Copyright 2024 The Bazel Authors. All rights reserved.
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
"""Supporting C++ compilation of generated code"""

load(":common/cc/cc_common.bzl", "cc_common")
load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/cc/cc_info.bzl", "CcInfo")

def get_feature_configuration(ctx, has_sources, extra_requested_features = []):
    """Returns C++ feature configuration for compiling and linking generated C++ files.

    Args:
        ctx: (RuleCtx) rule context.
        has_sources: (bool) Has the proto_library sources.
        extra_requested_features: (list[str]) Additionally requested features.
    Returns:
      (FeatureConfiguration) C++ feature configuration
    """
    cc_toolchain = cc_helper.find_cpp_toolchain(ctx)
    requested_features = ctx.features + extra_requested_features

    # TODO(bazel-team): Remove LAYERING_CHECK once we have verified that there are direct
    # dependencies for all generated #includes.
    unsupported_features = ctx.disabled_features + ["parse_headers", "layering_check"]
    if has_sources:
        requested_features.append("header_modules")
    else:
        unsupported_features.append("header_modules")
    return cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = requested_features,
        unsupported_features = unsupported_features,
    )

def _get_libraries_from_linking_outputs(linking_outputs, feature_configuration):
    library_to_link = linking_outputs.library_to_link
    if not library_to_link:
        return []
    outputs = []
    if library_to_link.static_library:
        outputs.append(library_to_link.static_library)
    if library_to_link.pic_static_library:
        outputs.append(library_to_link.pic_static_library)

    # On Windows, dynamic library is not built by default, so don't add them to files_to_build.
    if not cc_common.is_enabled(feature_configuration = feature_configuration, feature_name = "targets_windows"):
        if library_to_link.resolved_symlink_dynamic_library:
            outputs.append(library_to_link.resolved_symlink_dynamic_library)
        elif library_to_link.dynamic_library:
            outputs.append(library_to_link.dynamic_library)
        if library_to_link.resolved_symlink_interface_library:
            outputs.append(library_to_link.resolved_symlink_interface_library)
        elif library_to_link.interface_library:
            outputs.append(library_to_link.interface_library)
    return outputs

def cc_proto_compile_and_link(ctx, deps, sources, headers, disallow_dynamic_library = None, feature_configuration = None, alwayslink = False, **kwargs):
    """Creates C++ compilation and linking actions for C++ proto sources.

    Args:
        ctx: rule context
        deps: (list[CcInfo]) List of libraries to be added as dependencies to compilation and linking
            actions.
        sources:(list[File]) List of C++ sources files.
        headers: list(File] List of C++ headers files.
        disallow_dynamic_library: (bool) Are dynamic libraries disallowed.
        feature_configuration: (FeatureConfiguration) feature configuration to use.
        alwayslink: (bool) Should the library be always linked.
        **kwargs: Additional arguments passed to the compilation. See cc_common.compile.

    Returns:
        (CcInfo, list[File], list[File])
        - CcInfo provider with compilation context and linking context
        - A list of linked libraries related to this proto
        - A list of temporary files generated durind compilation
    """
    cc_toolchain = cc_helper.find_cpp_toolchain(ctx)
    feature_configuration = feature_configuration or get_feature_configuration(ctx, bool(sources))
    if disallow_dynamic_library == None:
        # TODO(dougk): Configure output artifact with action_config
        # once proto compile action is configurable from the crosstool.
        disallow_dynamic_library = not cc_common.is_enabled(
            feature_name = "supports_dynamic_linker",
            feature_configuration = feature_configuration,
        )

    (compilation_context, compilation_outputs) = cc_common.compile(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        srcs = sources,
        public_hdrs = headers,
        compilation_contexts = [dep[CcInfo].compilation_context for dep in deps],
        name = ctx.label.name,
        # Don't instrument the generated C++ files even when --collect_code_coverage is set.
        # If we actually start generating coverage instrumentation for .proto files based on coverage
        # data from the generated C++ files, this will have to be removed. Currently, the work done
        # to instrument those files and execute the instrumentation is all for nothing, and it can
        # be quite a bit of extra computation even when that's not made worse by performance bugs,
        # as in b/64963386.
        code_coverage_enabled = False,
        **kwargs
    )

    if sources:
        linking_context, linking_outputs = cc_common.create_linking_context_from_compilation_outputs(
            actions = ctx.actions,
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            compilation_outputs = compilation_outputs,
            linking_contexts = [dep[CcInfo].linking_context for dep in deps],
            name = ctx.label.name,
            disallow_dynamic_library = disallow_dynamic_library,
            alwayslink = alwayslink,
        )
        libraries = _get_libraries_from_linking_outputs(linking_outputs, feature_configuration)
    else:
        linking_context = cc_common.merge_linking_contexts(
            linking_contexts = [dep[CcInfo].linking_context for dep in deps if CcInfo in dep],
        )
        libraries = []

    return CcInfo(
        compilation_context = compilation_context,
        linking_context = linking_context,
        debug_context = cc_common.merge_debug_context(
            [cc_common.create_debug_context(compilation_outputs)] +
            [dep[CcInfo].debug_context() for dep in deps if CcInfo in dep],
        ),
    ), libraries, compilation_outputs.temps()