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
An alias that resolves the Windows x86_64 MSVC Rust toolchain for offline
distfile bootstrapping tests.

On Windows, Bazel itself is built with the hermetic MinGW LLVM toolchain
(`.bazelrc`), whereas `./compile.sh` ignores `.bazelrc` and bootstraps with the
host MSVC toolchain. Transitioning to `@rules_rs//rs/platforms:x86_64-pc-windows-msvc`
with `@local_config_cc_toolchains//:all` provides the MSVC Rust standard library
matching `rustc.exe`'s host triple without pulling the EULA-gated LLVM MSVC
toolchain.
"""

load("@with_cfg.bzl", "with_cfg")

windows_msvc_rust_toolchain_alias, _windows_msvc_rust_toolchain_alias = (
    with_cfg(native.alias)
        .set("platforms", [Label("@rules_rs//rs/platforms:x86_64-pc-windows-msvc")])
        .extend("extra_toolchains", ["@local_config_cc_toolchains//:all"])
        .build()
)
