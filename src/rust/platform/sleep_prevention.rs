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

//! Platform-specific sleep prevention for the Bazel/Blaze server.

#[cfg(any(target_os = "macos", target_os = "windows"))]
use std::num::NonZeroUsize;

#[cfg(target_os = "macos")]
mod sleep_prevention_darwin;
#[cfg(target_os = "macos")]
use sleep_prevention_darwin as imp;

#[cfg(target_os = "windows")]
mod sleep_prevention_windows;
#[cfg(target_os = "windows")]
use sleep_prevention_windows as imp;

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
mod sleep_prevention_unsupported;
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
use sleep_prevention_unsupported as imp;

/// Errors returned by sleep-prevention operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepPreventionError {
    /// Sleep prevention is not supported on this platform.
    Unsupported,
    /// `pop_disable_sleep` was called without a matching `push_disable_sleep`.
    Underflow,
}

/// Reference-counted stack managing a single active sleep-prevention guard.
///
/// Acquires a guard on the `0 -> 1` transition and drops it on the `1 -> 0`
/// transition.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub(crate) enum SleepStack<T> {
    Idle,
    Active { depth: NonZeroUsize, _guard: T },
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
impl<T> SleepStack<T> {
    /// Creates an idle stack with no active sleep-prevention guard.
    pub(crate) const fn new() -> Self {
        Self::Idle
    }

    /// Pushes a sleep-prevention request, invoking `acquire` when transitioning
    /// from idle to active.
    pub(crate) fn push(&mut self, acquire: impl FnOnce() -> T) {
        match self {
            Self::Idle => {
                *self = Self::Active { depth: NonZeroUsize::MIN, _guard: acquire() };
            }
            Self::Active { depth, .. } => {
                *depth = depth.checked_add(1).expect("sleep prevention stack depth overflow");
            }
        }
    }

    /// Pops a sleep-prevention request, dropping the active guard when
    /// transitioning from active back to idle.
    ///
    /// Returns [`SleepPreventionError::Underflow`] if the stack is already
    /// idle.
    pub(crate) fn pop(&mut self) -> Result<(), SleepPreventionError> {
        match self {
            Self::Idle => Err(SleepPreventionError::Underflow),
            Self::Active { depth, .. } => {
                if let Some(new_depth) = NonZeroUsize::new(depth.get() - 1) {
                    *depth = new_depth;
                } else {
                    *self = Self::Idle;
                }
                Ok(())
            }
        }
    }
}

/// Pushes a request to disable system sleep.
pub fn push_disable_sleep() -> Result<(), SleepPreventionError> {
    imp::push_disable_sleep()
}

/// Pops a request to disable system sleep.
pub fn pop_disable_sleep() -> Result<(), SleepPreventionError> {
    imp::pop_disable_sleep()
}
