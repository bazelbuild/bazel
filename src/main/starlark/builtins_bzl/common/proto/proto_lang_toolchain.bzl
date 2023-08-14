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

load(":common/proto/proto_info.bzl", "ProtoInfo")
load(":common/proto/proto_common.bzl", "ProtoLangToolchainInfo")
load(":common/proto/proto_semantics.bzl", "semantics")

def _rule_impl(ctx):
    provided_proto_sources = depset(transitive = [bp[ProtoInfo]._transitive_proto_sources for bp in ctx.attr.blacklisted_protos]).to_list()

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
            output_files = ctx.attr.output_files,
            plugin_format_flag = ctx.attr.plugin_format_flag,
            plugin = plugin,
            runtime = ctx.attr.runtime,
            provided_proto_sources = provided_proto_sources,
            proto_compiler = proto_compiler.files_to_run,
            protoc_opts = ctx.fragments.proto.experimental_protoc_opts,
            progress_message = ctx.attr.progress_message,
            mnemonic = ctx.attr.mnemonic,
            allowlist_different_package = ctx.attr.allowlist_different_package,
        ),
    ]

def make_proto_lang_toolchain(custom_proto_compiler):
    return rule(
        _rule_impl,
        attrs = dict(
            {
                "progress_message": attr.string(default = "Generating proto_library %{label}"),
                "mnemonic": attr.string(default = "GenProto"),
                "command_line": attr.string(mandatory = True),
                "output_files": attr.string(values = ["single", "multiple", "legacy"], default = "legacy"),
                "plugin_format_flag": attr.string(),
                "plugin": attr.label(
                    executable = True,
                    cfg = "exec",
                ),
                "runtime": attr.label(),
                "blacklisted_protos": attr.label_list(
                    providers = [ProtoInfo],
                ),
                "allowlist_different_package": attr.label(
                    default = semantics.allowlist_different_package,
                    cfg = "exec",
                    providers = ["PackageSpecificationProvider"],
                ),
            },
            **({
                "proto_compiler": attr.label(
                    cfg = "exec",
                    executable = True,
                ),
            } if custom_proto_compiler else {
                "_proto_compiler": attr.label(
                    cfg = "exec",
                    executable = True,
                    allow_files = True,
                    default = configuration_field("proto", "proto_compiler"),
                ),
            })
        ),
        provides = [ProtoLangToolchainInfo],
        fragments = ["proto"],
    )
