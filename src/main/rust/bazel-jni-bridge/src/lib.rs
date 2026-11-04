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

//! JNI C-ABI bridge allowing incremental in-tree delegation from `BazelServer_deploy.jar`
//! (`src/main/java/com/google/devtools/build/lib/...`) into Rust crates (`bazel-vfs`,
//! `bazel-cmdline`, `bazel-skyframe`, `bazel-util`).

use bazel_cmdline::Label;
use bazel_util::{md5, sha256};
use bazel_vfs::PathFragment;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Normalizes a path string using `bazel_vfs::PathFragment::create`.
/// Caller must free the returned string with `bazel_rs_free_string`.
#[no_mangle]
pub unsafe extern "C" fn bazel_rs_normalize_path(raw_path: *const c_char) -> *mut c_char {
    if raw_path.is_null() {
        return std::ptr::null_mut();
    }
    let Ok(s) = unsafe { CStr::from_ptr(raw_path) }.to_str() else {
        return std::ptr::null_mut();
    };
    let normalized = PathFragment::create(s);
    CString::new(normalized.path_string())
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

/// Parses and canonicalizes a Bazel label using `bazel_cmdline::Label::parse_canonical`.
/// Returns null if the label syntax is invalid.
#[no_mangle]
pub unsafe extern "C" fn bazel_rs_canonicalize_label(raw_label: *const c_char) -> *mut c_char {
    if raw_label.is_null() {
        return std::ptr::null_mut();
    }
    let Ok(s) = unsafe { CStr::from_ptr(raw_label) }.to_str() else {
        return std::ptr::null_mut();
    };
    let Ok(label) = Label::parse_canonical(s) else {
        return std::ptr::null_mut();
    };
    CString::new(label.canonical_form())
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

/// Computes the hex SHA-256 digest of a byte buffer (`data` of length `len`).
#[no_mangle]
pub unsafe extern "C" fn bazel_rs_sha256_hex(data: *const u8, len: usize) -> *mut c_char {
    let slice = if data.is_null() || len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    let hex = sha256::hex_digest(slice);
    CString::new(hex)
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

/// Computes the hex MD5 digest of a byte buffer (`data` of length `len`).
#[no_mangle]
pub unsafe extern "C" fn bazel_rs_md5_hex(data: *const u8, len: usize) -> *mut c_char {
    let slice = if data.is_null() || len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    let hex = md5::hex_digest(slice);
    CString::new(hex)
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

/// Frees a `CString` allocated by `bazel_rs_*` FFI functions.
#[no_mangle]
pub unsafe extern "C" fn bazel_rs_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_bridge_roundtrip() {
        let input = CString::new("//src/main/rust/../java:bazel").unwrap();
        let out_ptr = unsafe { bazel_rs_normalize_path(input.as_ptr()) };
        assert!(!out_ptr.is_null());
        let out_str = unsafe { CStr::from_ptr(out_ptr) }.to_str().unwrap();
        assert_eq!(out_str, "/src/main/java:bazel");
        unsafe { bazel_rs_free_string(out_ptr) };

        let label_in = CString::new("@rules_rs//rs:rust_binary.bzl").unwrap();
        let label_ptr = unsafe { bazel_rs_canonicalize_label(label_in.as_ptr()) };
        assert!(!label_ptr.is_null());
        let label_str = unsafe { CStr::from_ptr(label_ptr) }.to_str().unwrap();
        assert_eq!(label_str, "@@rules_rs//rs:rust_binary.bzl");
        unsafe { bazel_rs_free_string(label_ptr) };
    }
}

