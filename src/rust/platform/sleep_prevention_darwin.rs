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

//! macOS sleep prevention backed by IOKit power management assertions.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr::{self, NonNull};
use std::sync::Mutex;

use crate::{SleepPreventionError, SleepStack};

type CfAllocatorRef = *const c_void;
type CfStringRef = *const c_void;
type CfTypeRef = *const c_void;
type CfStringEncoding = u32;
type IoReturn = c_int;
type IopmAssertionId = u32;
type IopmAssertionLevel = u32;

const CF_STRING_ENCODING_UTF8: CfStringEncoding = 0x0800_0100;
const IO_RETURN_SUCCESS: IoReturn = 0;
const IOPM_ASSERTION_LEVEL_ON: IopmAssertionLevel = 255;
const IOPM_NULL_ASSERTION_ID: IopmAssertionId = 0;

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(
        alloc: CfAllocatorRef,
        c_str: *const c_char,
        encoding: CfStringEncoding,
    ) -> CfStringRef;
    fn CFRelease(cf: CfTypeRef);
}

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOPMAssertionCreateWithName(
        assertion_type: CfStringRef,
        assertion_level: IopmAssertionLevel,
        assertion_name: CfStringRef,
        assertion_id: *mut IopmAssertionId,
    ) -> IoReturn;
    // SAFETY: `IOPMAssertionRelease` takes a scalar `IOPMAssertionID` (`u32`)
    // by value with no pointer or aliasing invariants; IOKit validates the ID
    // and returns an `IOReturn` error if it is invalid or already released.
    safe fn IOPMAssertionRelease(assertion_id: IopmAssertionId) -> IoReturn;
}

unsafe extern "C" {
    // SAFETY: `mach_error_string` accepts any Mach/IOKit error code by value
    // and returns either null or a pointer to a static NUL-terminated C string.
    safe fn mach_error_string(error_value: IoReturn) -> *const c_char;
}

struct CfString(NonNull<c_void>);

impl CfString {
    fn from_c_str(s: &CStr) -> Option<Self> {
        // SAFETY: `s` is a valid NUL-terminated C string, `ptr::null()` selects
        // `kCFAllocatorDefault`, and any non-null return value carries an owned
        // +1 retain count.
        let raw =
            unsafe { CFStringCreateWithCString(ptr::null(), s.as_ptr(), CF_STRING_ENCODING_UTF8) };
        NonNull::new(raw.cast_mut()).map(Self)
    }

    fn as_ptr(&self) -> CfStringRef {
        self.0.as_ptr().cast_const()
    }
}

impl Drop for CfString {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a valid non-null `CFStringRef` with an owned +1
        // retain count acquired in `CfString::from_c_str`.
        unsafe { CFRelease(self.as_ptr()) };
    }
}

struct SleepAssertion(IopmAssertionId);

impl SleepAssertion {
    fn acquire() -> Option<Self> {
        let (Some(assertion_type), Some(reason)) =
            (CfString::from_c_str(c"NoIdleSleepAssertion"), CfString::from_c_str(c"build.bazel"))
        else {
            eprintln!("CFStringCreateWithCString failed while creating sleep assertion");
            return None;
        };

        let mut assertion_id: IopmAssertionId = IOPM_NULL_ASSERTION_ID;
        // SAFETY: `assertion_type` and `reason` are valid owned `CFStringRef`s
        // that outlive this call, and `assertion_id` is a valid writable
        // pointer.
        let status = unsafe {
            IOPMAssertionCreateWithName(
                assertion_type.as_ptr(),
                IOPM_ASSERTION_LEVEL_ON,
                reason.as_ptr(),
                &mut assertion_id,
            )
        };

        if status != IO_RETURN_SUCCESS {
            eprintln!("IOPMAssertionCreateWithName failed: {}", format_mach_error(status));
            return None;
        }
        (assertion_id != IOPM_NULL_ASSERTION_ID).then_some(Self(assertion_id))
    }
}

impl Drop for SleepAssertion {
    fn drop(&mut self) {
        let status = IOPMAssertionRelease(self.0);
        if status != IO_RETURN_SUCCESS {
            eprintln!("IOPMAssertionRelease failed: {}", format_mach_error(status));
        }
    }
}

static SLEEP_STATE: Mutex<SleepStack<Option<SleepAssertion>>> = Mutex::new(SleepStack::new());

fn format_mach_error(err: IoReturn) -> String {
    let msg = mach_error_string(err);
    if msg.is_null() {
        return format!("({err})");
    }
    // SAFETY: `msg` was verified non-null and points to a valid static C string.
    let msg = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    format!("{msg} ({err})")
}

pub fn push_disable_sleep() -> Result<(), SleepPreventionError> {
    let mut state = SLEEP_STATE.lock().unwrap_or_else(|e| e.into_inner());
    state.push(SleepAssertion::acquire);
    Ok(())
}

pub fn pop_disable_sleep() -> Result<(), SleepPreventionError> {
    let mut state = SLEEP_STATE.lock().unwrap_or_else(|e| e.into_inner());
    state.pop()
}
