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

"""Bazel providers for proto rules."""

ProtoLangToolchainInfo = provider(
    doc = "Specifies how to generate language-specific code from .proto files. Used by LANG_proto_library rules.",
    fields = dict(
        out_replacement_format_flag = "(str) Format string used when passing output to the plugin used by proto compiler.",
        plugin_format_flag = "(str) Format string used when passing plugin to proto compiler.",
        plugin = "(FilesToRunProvider) Proto compiler plugin.",
        runtime = "(Target) Runtime.",
        provided_proto_sources = "(list[ProtoSource]) Proto sources provided by the toolchain.",
        proto_compiler = "(FilesToRunProvider) Proto compiler.",
        protoc_opts = "(list[str]) Options to pass to proto compiler.",
        progress_message = "(str) Progress message to set on the proto compiler action.",
        mnemonic = "(str) Mnemonic to set on the proto compiler action.",
    ),
)
