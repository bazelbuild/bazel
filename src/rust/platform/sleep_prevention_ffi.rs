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

//! C FFI bindings for `sleep_prevention`.

use sleep_prevention::SleepPreventionError;

const STATUS_OK: i32 = 0;
const STATUS_UNSUPPORTED: i32 = -1;
const STATUS_UNDERFLOW: i32 = -2;

fn status_code(result: Result<(), SleepPreventionError>) -> i32 {
    match result {
        Ok(()) => STATUS_OK,
        Err(SleepPreventionError::Unsupported) => STATUS_UNSUPPORTED,
        Err(SleepPreventionError::Underflow) => STATUS_UNDERFLOW,
    }
}

/// Pushes a request to disable automatic system sleep.
///
/// Requests are reference-counted and thread-safe; each call must be paired
/// with a matching [`bazel_sleep_prevention_pop`] call. On supported platforms
/// (macOS and Windows), OS sleep prevention is enabled when the count
/// transitions from 0 to 1. Any OS-level acquisition failure is logged to
/// stderr rather than failing the call.
///
/// Returns:
/// - `0` on success.
/// - `-1` if sleep prevention is unsupported on the current platform.
#[unsafe(no_mangle)]
pub extern "C" fn bazel_sleep_prevention_push() -> i32 {
    status_code(sleep_prevention::push_disable_sleep())
}

/// Pops a request to disable automatic system sleep.
///
/// Must match an earlier [`bazel_sleep_prevention_push`] call. On supported
/// platforms (macOS and Windows), OS sleep prevention is released when the
/// count transitions from 1 to 0. Any OS-level release failure is logged to
/// stderr rather than failing the call.
///
/// Returns:
/// - `0` on success.
/// - `-1` if sleep prevention is unsupported on the current platform.
/// - `-2` if called without a matching push (underflow).
#[unsafe(no_mangle)]
pub extern "C" fn bazel_sleep_prevention_pop() -> i32 {
    status_code(sleep_prevention::pop_disable_sleep())
}
