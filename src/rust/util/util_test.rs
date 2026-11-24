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

use std::env;
use std::process::Command;

use googletest::{expect_true, gtest, verify_eq, Result};

#[gtest]
fn is_running_within_test_checks_test_tmpdir() -> Result<()> {
    const CHILD_EXPECTATION_ENV: &str = "BAZEL_UTIL_TEST_CHILD_EXPECT";
    if let Some(expected) = env::var_os(CHILD_EXPECTATION_ENV) {
        verify_eq!(util::is_running_within_test(), expected == "1")?;
        return Ok(());
    }

    // Re-execute the test binary in a child process with a controlled
    // `TEST_TMPDIR` value instead of mutating the environment in-process:
    // `std::env::{set_var, remove_var}` are `unsafe` in Rust 2024 because
    // modifying process-wide env vars races with concurrent reads on the
    // test harness's worker threads.
    let exe = env::current_exe()?;
    for (tmpdir, expected) in [(None, "0"), (Some(""), "1"), (Some("/tmp/test"), "1")] {
        let mut command = Command::new(&exe);
        command
            .arg("--exact")
            .arg("is_running_within_test_checks_test_tmpdir")
            .env(CHILD_EXPECTATION_ENV, expected);
        match tmpdir {
            Some(value) => {
                command.env("TEST_TMPDIR", value);
            }
            None => {
                command.env_remove("TEST_TMPDIR");
            }
        }
        let status = command.status()?;
        expect_true!(status.success(), "child test failed for TEST_TMPDIR={tmpdir:?}");
    }
    Ok(())
}
