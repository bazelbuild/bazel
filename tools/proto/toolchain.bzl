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

_use_proto_toolchain_from_rules_proto = "use_proto_toolchain_from_rules_proto_do_not_use_or_we_will_break_you_without_mercy"

def maybe_register_proto_toolchain():
    if getattr(_proto_common, _use_proto_toolchain_from_rules_proto, False):
        # --incompatible_use_proto_toolchain_from_rules_proto has been flipped,
        # nothing to do.
        return

    # buildifier: disable=native-proto
    native.proto_toolchain(
        name = "default_toolchain",
        # Must be earlier than the first version of rules_proto that defined
        # a proto toolchain (0.1.0).
        rules_proto_version = "0.0.1",
        tags = [
            "__PROTO_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__",
        ],
    )

    native.toolchain(
        name = "toolchain",
        toolchain = ":default_toolchain",
        toolchain_type = "@bazel_tools//tools/proto:toolchain_type",
    )
