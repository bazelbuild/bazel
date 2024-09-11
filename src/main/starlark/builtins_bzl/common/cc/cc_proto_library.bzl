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
"""Starlark implementation of cc_proto_library"""

load(":common/cc/cc_common.bzl", "cc_common")
load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/cc/cc_info.bzl", "CcInfo")
load(":common/cc/semantics.bzl", "semantics")
load(":common/proto/proto_common.bzl", "toolchains", proto_common = "proto_common_do_not_use")
load(":common/proto/proto_info.bzl", "ProtoInfo")

_ProtoCcFilesInfo = provider(fields = ["files"], doc = "Provide cc proto files.")
_ProtoCcHeaderInfo = provider(fields = ["headers"], doc = "Provide cc proto headers.")

def _get_feature_configuration(ctx, cc_toolchain, proto_info, should_generate_code):
    requested_features = list(ctx.features)
    unsupported_features = ctx.disabled_features + ["parse_headers", "layering_check"]
    if should_generate_code or len(proto_info.direct_sources) == 0:
        unsupported_features.append("header_modules")
    else:
        requested_features.append("header_modules")
    return cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = requested_features,
        unsupported_features = unsupported_features,
    )

def _get_output_files(actions, proto_info, suffixes):
    result = []
    for suffix in suffixes:
        result.extend(proto_common.declare_generated_files(
            actions = actions,
            proto_info = proto_info,
            extension = suffix,
        ))
    return result

# TODO(b/364873432): Make this code actually work.
def _get_strip_include_prefix(ctx, proto_info):
    proto_root = proto_info.proto_source_root
    if proto_root == "." or proto_root == ctx.label.workspace_root:
        return ""
    strip_include_prefix = ""
    if proto_root.startswith(ctx.bin_dir.path):
        proto_root = proto_root[len(ctx.bin_dir.path) + 1:]
    elif proto_root.startswith(ctx.genfiles_dir.path):
        proto_root = proto_root[len(ctx.genfiles_dir.path) + 1:]

    if proto_root.startswith(ctx.label.workspace_root):
        proto_root = proto_root[len(ctx.label.workspace_root):]

    strip_include_prefix = "//" + proto_root
    return strip_include_prefix

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

def _aspect_impl(target, ctx):
    proto_info = target[ProtoInfo]
    proto_configuration = ctx.fragments.proto

    sources = []
    headers = []
    textual_hdrs = []
    additional_exported_hdrs = []

    proto_toolchain = toolchains.find_toolchain(ctx, "_aspect_cc_proto_toolchain", semantics.CC_PROTO_TOOLCHAIN)
    should_generate_code = proto_common.experimental_should_generate_code(proto_info, proto_toolchain, "cc_proto_library", target.label)

    if should_generate_code:
        if len(proto_info.direct_sources) != 0:
            sources = _get_output_files(
                ctx.actions,
                proto_info,
                proto_configuration.cc_proto_library_source_suffixes(),
            )
            headers = _get_output_files(
                ctx.actions,
                proto_info,
                proto_configuration.cc_proto_library_header_suffixes(),
            )
            header_provider = _ProtoCcHeaderInfo(headers = depset(headers))
        else:
            # If this proto_library doesn't have sources, it provides the combined headers of all its
            # direct dependencies. Thus, if a direct dependency does have sources, the generated files
            # are also provided by this library. If a direct dependency does not have sources, it will
            # do the same thing, so that effectively this library looks through all source-less
            # proto_libraries and provides all generated headers of the proto_libraries with sources
            # that it depends on.
            transitive_headers = []
            for dep in getattr(ctx.rule.attr, "deps", []):
                if _ProtoCcHeaderInfo in dep:
                    textual_hdrs.extend(dep[_ProtoCcHeaderInfo].headers.to_list())
                    transitive_headers.append(dep[_ProtoCcHeaderInfo].headers)
            header_provider = _ProtoCcHeaderInfo(headers = depset(transitive = transitive_headers))

    else:  # shouldn't generate code
        # Hack: This is a proto_library for descriptor.proto or similar.
        #
        # The headers of those libraries are precomputed. They are also explicitly part of normal
        # cc_library rules that export them in their 'hdrs' attribute, and compile them as header
        # module if requested.
        #
        # The sole purpose of a proto_library with forbidden srcs is so other proto_library rules
        # can import them from a protocol buffer, as proto_library rules can only depend on other
        # proto library rules.
        for source in proto_info.direct_sources:
            for suffix in proto_configuration.cc_proto_library_header_suffixes():
                # We add the header to the proto_library's module map as additional (textual) header for
                # two reasons:
                # 1. The header will be exported via a normal cc_library, and a header must only be exported
                #    non-textually from one library.
                # 2. We want to allow proto_library rules that depend on the bootstrap-hack proto_library
                #    to be layering-checked; we need to provide a module map for the layering check to work.
                additional_exported_hdrs.append(source.short_path[:-len(source.extension)] + suffix)
        header_provider = _ProtoCcHeaderInfo(headers = depset())

    proto_common.compile(
        actions = ctx.actions,
        proto_info = proto_info,
        proto_lang_toolchain_info = proto_toolchain,
        generated_files = sources + headers,
        experimental_output_files = "multiple",
    )

    deps = ([proto_toolchain.runtime] if proto_toolchain.runtime else []) + getattr(ctx.rule.attr, "deps", [])

    cc_toolchain = cc_helper.find_cpp_toolchain(ctx)
    feature_configuration = _get_feature_configuration(ctx, cc_toolchain, proto_info, should_generate_code)
    (compilation_context, compilation_outputs) = cc_common.compile(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        srcs = sources,
        public_hdrs = headers,
        compilation_contexts = [dep[CcInfo].compilation_context for dep in deps],
        name = ctx.label.name,
        # Don't instrument the generated C++ files even when --collect_code_coverage is set.
        code_coverage_enabled = False,
        additional_exported_hdrs = additional_exported_hdrs,
        textual_hdrs = textual_hdrs,
        strip_include_prefix = _get_strip_include_prefix(ctx, proto_info),
    )

    if sources:
        # TODO(dougk): Configure output artifact with action_config
        # once proto compile action is configurable from the crosstool.
        disallow_dynamic_library = not cc_common.is_enabled(
            feature_name = "supports_dynamic_linker",
            feature_configuration = feature_configuration,
        )
        linking_context, linking_outputs = cc_common.create_linking_context_from_compilation_outputs(
            actions = ctx.actions,
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            compilation_outputs = compilation_outputs,
            linking_contexts = [dep[CcInfo].linking_context for dep in deps],
            name = ctx.label.name,
            disallow_dynamic_library = disallow_dynamic_library,
            test_only_target = getattr(ctx.rule.attr, "testonly", False),
        )
        libraries = _get_libraries_from_linking_outputs(linking_outputs, feature_configuration)
    else:
        linking_context = cc_common.merge_linking_contexts(
            linking_contexts = [dep[CcInfo].linking_context for dep in deps if CcInfo in dep],
        )
        libraries = []

    return [
        CcInfo(
            compilation_context = compilation_context,
            linking_context = linking_context,
            debug_context = cc_common.merge_debug_context(
                [cc_common.create_debug_context(compilation_outputs)] +
                [dep[CcInfo].debug_context() for dep in deps if CcInfo in dep],
            ),
        ),
        _ProtoCcFilesInfo(files = depset(sources + headers + libraries)),
        OutputGroupInfo(temp_files_INTERNAL_ = compilation_outputs.temps()),
        header_provider,
    ]

cc_proto_aspect = aspect(
    implementation = _aspect_impl,
    attr_aspects = ["deps"],
    fragments = ["cpp", "proto"],
    required_providers = [ProtoInfo],
    provides = [CcInfo],
    attrs = toolchains.if_legacy_toolchain({"_aspect_cc_proto_toolchain": attr.label(
        default = configuration_field(fragment = "proto", name = "proto_toolchain_for_cc"),
    )}),
    toolchains = cc_helper.use_cpp_toolchain() + toolchains.use_toolchain(semantics.CC_PROTO_TOOLCHAIN),
)

def _cc_proto_library_rule(ctx):
    if len(ctx.attr.deps) != 1:
        fail(
            "'deps' attribute must contain exactly one label " +
            "(we didn't name it 'dep' for consistency). " +
            "The main use-case for multiple deps is to create a rule that contains several " +
            "other targets. This makes dependency bloat more likely. It also makes it harder" +
            "to remove unused deps.",
            attr = "deps",
        )
    dep = ctx.attr.deps[0]

    proto_toolchain = toolchains.find_toolchain(ctx, "_aspect_cc_proto_toolchain", semantics.CC_PROTO_TOOLCHAIN)
    proto_common.check_collocated(ctx.label, dep[ProtoInfo], proto_toolchain)

    return [DefaultInfo(files = dep[_ProtoCcFilesInfo].files), dep[CcInfo], dep[OutputGroupInfo]]

cc_proto_library = rule(
    implementation = _cc_proto_library_rule,
    doc = """
<p>
<code>cc_proto_library</code> generates C++ code from <code>.proto</code> files.
</p>

<p>
<code>deps</code> must point to <a href="protocol-buffer.html#proto_library"><code>proto_library
</code></a> rules.
</p>

<p>
Example:
</p>

<pre>
<code class="lang-starlark">
cc_library(
    name = "lib",
    deps = [":foo_cc_proto"],
)

cc_proto_library(
    name = "foo_cc_proto",
    deps = [":foo_proto"],
)

proto_library(
    name = "foo_proto",
)
</code>
</pre>
""",
    attrs = {
        "deps": attr.label_list(
            aspects = [cc_proto_aspect],
            allow_rules = ["proto_library"],
            allow_files = False,
            doc = """
The list of <a href="protocol-buffer.html#proto_library"><code>proto_library</code></a>
rules to generate C++ code for.""",
        ),
    } | toolchains.if_legacy_toolchain({
        "_aspect_cc_proto_toolchain": attr.label(
            default = configuration_field(fragment = "proto", name = "proto_toolchain_for_cc"),
        ),
    }),
    provides = [CcInfo],
    toolchains = toolchains.use_toolchain(semantics.CC_PROTO_TOOLCHAIN),
)
