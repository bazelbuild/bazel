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

"""A Starlark implementation of the proto_lang_toolchain rule."""

load(":common/proto/providers.bzl", "ProtoLangToolchainInfo")
load(":common/proto/proto_semantics.bzl", "semantics")
load(":common/rule_util.bzl", "merge_attrs")

ProtoInfo = _builtins.toplevel.ProtoInfo
proto_common = _builtins.toplevel.proto_common

def _rule_impl(ctx):
    provided_proto_sources = []
    transitive_files = depset(transitive = [bp[ProtoInfo].transitive_sources for bp in ctx.attr.blacklisted_protos])
    for file in transitive_files.to_list():
        source_root = file.root.path
        provided_proto_sources.append(proto_common.ProtoSource(file, file, source_root))

    flag = ctx.attr.command_line
    if flag.find("$(PLUGIN_OUT)") > -1:
        fail("in attribute 'command_line': Placeholder '$(PLUGIN_OUT)' is not supported.")
    flag = flag.replace("$(OUT)", "%s")

    plugin = None
    if ctx.attr.plugin != None:
        plugin = ctx.attr.plugin[DefaultInfo].files_to_run

    proto_compiler = getattr(ctx.attr, "proto_compiler", None)
    proto_compiler = getattr(ctx.attr, "_proto_compiler", proto_compiler)

    return [
        DefaultInfo(
            files = depset(),
            runfiles = ctx.runfiles(),
        ),
        ProtoLangToolchainInfo(
            out_replacement_format_flag = flag,
            plugin_format_flag = ctx.attr.plugin_format_flag,
            plugin = plugin,
            runtime = ctx.attr.runtime,
            provided_proto_sources = provided_proto_sources,
            proto_compiler = proto_compiler.files_to_run,
            protoc_opts = ctx.fragments.proto.experimental_protoc_opts,
            progress_message = ctx.attr.progress_message,
            mnemonic = ctx.attr.mnemonic,
        ),
    ]

proto_lang_toolchain_attrs = {
    "progress_message": attr.string(default = "Generating proto_library %{label}"),
    "mnemonic": attr.string(default = "GenProto"),
    "command_line": attr.string(mandatory = True),
    "plugin_format_flag": attr.string(),
    "plugin": attr.label(
        executable = True,
        cfg = "exec",
        allow_files = True,
    ),
    "runtime": attr.label(
        allow_files = True,
    ),
    "blacklisted_protos": attr.label_list(
        allow_files = True,
        providers = [ProtoInfo],
    ),
}

proto_lang_toolchain_custom_protoc = rule(
    implementation = _rule_impl,
    attrs = merge_attrs(
        proto_lang_toolchain_attrs,
        {
            "proto_compiler": attr.label(
                cfg = "exec",
                executable = True,
                allow_files = True,
            ),
        },
    ),
    provides = [ProtoLangToolchainInfo],
    fragments = ["proto"] + semantics.EXTRA_FRAGMENTS,
)

proto_lang_toolchain_default_protoc = rule(
    implementation = _rule_impl,
    attrs = merge_attrs(
        proto_lang_toolchain_attrs,
        {
            "_proto_compiler": attr.label(
                cfg = "exec",
                executable = True,
                allow_files = True,
                default = configuration_field("proto", "proto_compiler"),
            ),
        },
    ),
    provides = [ProtoLangToolchainInfo],
    fragments = ["proto"] + semantics.EXTRA_FRAGMENTS,
)

def proto_lang_toolchain(
        name = None,
        proto_compiler = None,
        **kwargs):
    if proto_compiler != None:
        proto_lang_toolchain_custom_protoc(
            name = name,
            proto_compiler = proto_compiler,
            **kwargs
        )
    else:
        proto_lang_toolchain_default_protoc(
            name = name,
            **kwargs
        )
