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

"""Server Rust FFI crates linked into Bazel."""

# Server Rust FFI crates exposed to the Java server via FFM (SkymetalBridge).
# Adding a crate here links it into the server and exports all of its `bazel_*`
# symbols on Linux, macOS, and Windows.
RUST_FFI_DEPS = [
    # Installs TrackingAllocator as the #[global_allocator] for all Rust code in
    # every binary that links these crates.
    "//src/rust/common:tracking_allocator_ffi",
    "//src/rust/platform:sleep_prevention_ffi",
]
