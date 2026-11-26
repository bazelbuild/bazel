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

//! Windows sleep prevention backed by `SetThreadExecutionState`.

use std::sync::Mutex;

use crate::{SleepPreventionError, SleepStack};

const ES_SYSTEM_REQUIRED: u32 = 0x0000_0001;
const ES_AWAYMODE_REQUIRED: u32 = 0x0000_0040;
const ES_CONTINUOUS: u32 = 0x8000_0000;

unsafe extern "system" {
    // SAFETY: `SetThreadExecutionState` takes only a bitmask of execution state
    // flags by value and has no pointer or aliasing invariants.
    safe fn SetThreadExecutionState(es_flags: u32) -> u32;
}

struct SleepGuard;

impl SleepGuard {
    fn acquire() -> Self {
        SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED);
        Self
    }
}

impl Drop for SleepGuard {
    fn drop(&mut self) {
        SetThreadExecutionState(ES_CONTINUOUS);
    }
}

static SLEEP_STATE: Mutex<SleepStack<SleepGuard>> = Mutex::new(SleepStack::new());

pub fn push_disable_sleep() -> Result<(), SleepPreventionError> {
    let mut state = SLEEP_STATE.lock().unwrap_or_else(|e| e.into_inner());
    state.push(SleepGuard::acquire);
    Ok(())
}

pub fn pop_disable_sleep() -> Result<(), SleepPreventionError> {
    let mut state = SLEEP_STATE.lock().unwrap_or_else(|e| e.into_inner());
    state.pop()
}
