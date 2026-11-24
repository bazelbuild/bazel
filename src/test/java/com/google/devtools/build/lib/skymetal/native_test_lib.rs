// Copyright 2026 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Native C-ABI test helpers for `SkymetalBridgeTest`.

/// Invokes the C-ABI callback `cb`.
// SAFETY: `bazel_test_invoke_callback` is unique in the linked test shared library and no other
// object defines a symbol with this name.
#[unsafe(no_mangle)]
pub extern "C" fn bazel_test_invoke_callback(cb: extern "C" fn()) {
    cb();
}

/// Sums arguments of mixed primitive and pointer types.
// SAFETY: `bazel_test_mixed_types` is unique in the linked test shared library and no other
// object defines a symbol with this name.
#[unsafe(no_mangle)]
pub extern "C" fn bazel_test_mixed_types(i: i32, j: i64, d: f64, p: &i32) -> i64 {
    i64::from(i) + j + (d as i64) + i64::from(*p)
}
