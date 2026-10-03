# Copyright 2026 The Bazel Authors. All rights reserved.
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
An alias that cross-compiles its target for Windows with the hermetic LLVM
toolchains, regardless of the host platform.

This is equivalent to building the target with
`--config=llvm --platforms=@llvm//platforms:windows_x86_64`.
"""

load("@with_cfg.bzl", "with_cfg")

# Keep the toolchains in sync with the `llvm` config in .bazelrc.
windows_llvm_alias, _windows_llvm_alias = (
    with_cfg(native.alias)
        .set("platforms", [Label("@llvm//platforms:windows_x86_64")])
        .extend("extra_toolchains", ["@llvm//toolchain:all"])
        .build()
)
