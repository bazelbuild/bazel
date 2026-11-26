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

use googletest::{expect_eq, gtest};
use sleep_prevention::{pop_disable_sleep, push_disable_sleep, SleepPreventionError};

#[gtest]
fn push_and_pop_return_platform_status() {
    let expected = if cfg!(any(target_os = "macos", target_os = "windows")) {
        Ok(())
    } else {
        Err(SleepPreventionError::Unsupported)
    };
    expect_eq!(push_disable_sleep(), expected);
    expect_eq!(pop_disable_sleep(), expected);

    // Verify nested push/pop and re-enabling after returning to zero.
    expect_eq!(push_disable_sleep(), expected);
    expect_eq!(push_disable_sleep(), expected);
    expect_eq!(pop_disable_sleep(), expected);
    expect_eq!(pop_disable_sleep(), expected);

    if cfg!(any(target_os = "macos", target_os = "windows")) {
        expect_eq!(pop_disable_sleep(), Err(SleepPreventionError::Underflow));
    }
}
