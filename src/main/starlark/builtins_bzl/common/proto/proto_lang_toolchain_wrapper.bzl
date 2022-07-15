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

"""Macro encapsulating the proto_lang_toolchain implementation.

This is needed since proto compiler can be defined, or used as a default one.
There are two implementations of proto_lang_toolchain - one with public proto_compiler attribute, and the other one with private compiler.
"""

load(":common/proto/proto_lang_toolchain_default_protoc.bzl", toolchain_default_protoc = "proto_lang_toolchain")
load(":common/proto/proto_lang_toolchain_custom_protoc.bzl", toolchain_custom_protoc = "proto_lang_toolchain")

def proto_lang_toolchain(
        proto_compiler = None,
        **kwargs):
    if proto_compiler != None:
        toolchain_custom_protoc(
            proto_compiler = proto_compiler,
            **kwargs
        )
    else:
        toolchain_default_protoc(
            **kwargs
        )
