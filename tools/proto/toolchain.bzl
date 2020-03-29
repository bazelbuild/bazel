# Copyright 2020 The Bazel Authors. All rights reserved.
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

# buildifier: disable=native-proto
_proto_common = proto_common

_provide_proto_toolchain_in_tools_workspace = "provide_proto_toolchain_in_tools_workspace_do_not_use_or_we_will_break_you_without_mercy"

def maybe_register_proto_toolchain():
    if not hasattr(_proto_common, _provide_proto_toolchain_in_tools_workspace):
        # --incompatible_provide_proto_toolchain_in_tools_workspace has been
        # flipped, nothing to do.
        return

    native.register_toolchains("@bazel_tools//tools/proto:toolchain")

def maybe_create_proto_toolchain_targets():
    if not hasattr(_proto_common, _provide_proto_toolchain_in_tools_workspace):
        # --incompatible_provide_proto_toolchain_in_tools_workspace has been
        # flipped, nothing to do.
        return

    # buildifier: disable=native-proto
    native.proto_toolchain(
        name = "default_toolchain",
        tags = [
            "__PROTO_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__",
        ],
    )

    native.toolchain(
        name = "toolchain",
        toolchain = ":default_toolchain",
        toolchain_type = "@bazel_tools//tools/proto:toolchain_type",
    )
