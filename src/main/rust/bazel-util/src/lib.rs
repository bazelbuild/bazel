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

//! Core Bazel utilities (`ExitCode`, `DetailedExitCode`, `Os`, `Clock`, `Md5`, `Sha256`).

use bazel_proto::failure_details::FailureDetail;
use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Exit codes shared by Bazel client and server, matching `ExitCode.java`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExitCode {
    pub code: i32,
    pub name: &'static str,
    pub infrastructure_failure: bool,
}

impl ExitCode {
    pub const SUCCESS: Self = Self::new(0, "SUCCESS", false);
    pub const BUILD_FAILURE: Self = Self::new(1, "BUILD_FAILURE", false);
    pub const PARSING_FAILURE: Self = Self::new(1, "PARSING_FAILURE", false);
    pub const COMMAND_LINE_ERROR: Self = Self::new(2, "COMMAND_LINE_ERROR", false);
    pub const TESTS_FAILED: Self = Self::new(3, "TESTS_FAILED", false);
    pub const PARTIAL_ANALYSIS_FAILURE: Self = Self::new(3, "PARTIAL_ANALYSIS_FAILURE", false);
    pub const NO_TESTS_FOUND: Self = Self::new(4, "NO_TESTS_FOUND", false);
    pub const RUN_FAILURE: Self = Self::new(6, "RUN_FAILURE", false);
    pub const ANALYSIS_FAILURE: Self = Self::new(7, "ANALYSIS_FAILURE", false);
    pub const INTERRUPTED: Self = Self::new(8, "INTERRUPTED", false);
    pub const LOCK_HELD_NOBLOCK_FOR_LOCK: Self = Self::new(9, "LOCK_HELD_NOBLOCK_FOR_LOCK", false);
    pub const REMOTE_ENVIRONMENTAL_ERROR: Self = Self::new(32, "REMOTE_ENVIRONMENTAL_ERROR", true);
    pub const OOM_ERROR: Self = Self::new(33, "OOM_ERROR", true);
    pub const REMOTE_ERROR: Self = Self::new(34, "REMOTE_ERROR", true);
    pub const LOCAL_ENVIRONMENTAL_ERROR: Self = Self::new(36, "LOCAL_ENVIRONMENTAL_ERROR", true);
    pub const BLAZE_INTERNAL_ERROR: Self = Self::new(37, "BLAZE_INTERNAL_ERROR", true);
    pub const PUBLISH_ERROR: Self = Self::new(38, "PUBLISH_ERROR", true);
    pub const REMOTE_CACHE_EVICTED: Self = Self::new(39, "REMOTE_CACHE_EVICTED", true);
    pub const PERSISTENT_BUILD_EVENT_SERVICE_UPLOAD_ERROR: Self = Self::new(
        45,
        "PERSISTENT_BUILD_EVENT_SERVICE_UPLOAD_ERROR",
        false,
    );
    pub const EXTERNAL_DEPS_ERROR: Self = Self::new(48, "EXTERNAL_DEPS_ERROR", false);

    pub const fn new(code: i32, name: &'static str, infrastructure_failure: bool) -> Self {
        Self {
            code,
            name,
            infrastructure_failure,
        }
    }

    pub const fn numeric_exit_code(&self) -> i32 {
        self.code
    }
}

impl fmt::Display for ExitCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.code)
    }
}

/// Combines an [`ExitCode`] with an optional [`FailureDetail`] proto message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetailedExitCode {
    pub exit_code: ExitCode,
    pub failure_detail: Option<FailureDetail>,
}

impl DetailedExitCode {
    pub const fn success() -> Self {
        Self {
            exit_code: ExitCode::SUCCESS,
            failure_detail: None,
        }
    }

    pub fn of(exit_code: ExitCode, failure_detail: FailureDetail) -> Self {
        Self {
            exit_code,
            failure_detail: Some(failure_detail),
        }
    }

    pub fn from_exit_code(exit_code: ExitCode) -> Self {
        Self {
            exit_code,
            failure_detail: None,
        }
    }

    pub fn is_success(&self) -> bool {
        self.exit_code == ExitCode::SUCCESS
    }
}

/// Host operating system enumeration matching `com.google.devtools.build.lib.util.OS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Os {
    Darwin,
    Linux,
    Windows,
    FreeBsd,
    OpenBsd,
    Unknown,
}

impl Os {
    pub const fn current() -> Self {
        if cfg!(target_os = "macos") {
            Self::Darwin
        } else if cfg!(target_os = "linux") {
            Self::Linux
        } else if cfg!(target_os = "windows") {
            Self::Windows
        } else if cfg!(target_os = "freebsd") {
            Self::FreeBsd
        } else if cfg!(target_os = "openbsd") {
            Self::OpenBsd
        } else {
            Self::Unknown
        }
    }

    pub const fn canonical_name(&self) -> &'static str {
        match self {
            Self::Darwin => "osx",
            Self::Linux => "linux",
            Self::Windows => "windows",
            Self::FreeBsd => "freebsd",
            Self::OpenBsd => "openbsd",
            Self::Unknown => "unknown",
        }
    }
}

/// Abstract clock interface matching `com.google.devtools.build.lib.clock.Clock`.
pub trait Clock: Send + Sync {
    fn current_time_millis(&self) -> u64;
    fn nano_time(&self) -> u128;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct JavaClock;

impl Clock for JavaClock {
    fn current_time_millis(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64
    }

    fn nano_time(&self) -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos()
    }
}

/// RFC 1321 MD5 implementation (used by Bazel to compute `$OUTPUT_BASE` directory names
/// from the workspace directory path and for `Fingerprint`).
pub mod md5 {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];

    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    pub fn digest(data: &[u8]) -> [u8; 16] {
        let mut a0: u32 = 0x67452301;
        let mut b0: u32 = 0xefcdab89;
        let mut c0: u32 = 0x98badcfe;
        let mut d0: u32 = 0x10325476;

        let bit_len = (data.len() as u64).wrapping_mul(8);
        let mut msg = data.to_vec();
        msg.push(0x80);
        while (msg.len() % 64) != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&bit_len.to_le_bytes());

        for chunk in msg.chunks_exact(64) {
            let mut m = [0u32; 16];
            for (i, word) in chunk.chunks_exact(4).enumerate() {
                m[i] = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            }

            let mut a = a0;
            let mut b = b0;
            let mut c = c0;
            let mut d = d0;

            for i in 0..64 {
                let (f, g) = match i {
                    0..=15 => ((b & c) | ((!b) & d), i),
                    16..=31 => ((d & b) | ((!d) & c), (5 * i + 1) % 16),
                    32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                    _ => (c ^ (b | (!d)), (7 * i) % 16),
                };
                let temp = d;
                d = c;
                c = b;
                b = b.wrapping_add(
                    a.wrapping_add(f)
                        .wrapping_add(K[i])
                        .wrapping_add(m[g])
                        .rotate_left(S[i]),
                );
                a = temp;
            }

            a0 = a0.wrapping_add(a);
            b0 = b0.wrapping_add(b);
            c0 = c0.wrapping_add(c);
            d0 = d0.wrapping_add(d);
        }

        let mut out = [0u8; 16];
        out[0..4].copy_from_slice(&a0.to_le_bytes());
        out[4..8].copy_from_slice(&b0.to_le_bytes());
        out[8..12].copy_from_slice(&c0.to_le_bytes());
        out[12..16].copy_from_slice(&d0.to_le_bytes());
        out
    }

    pub fn hex_digest(data: &[u8]) -> String {
        super::to_hex(&digest(data))
    }
}

/// FIPS 180-4 SHA-256 implementation for content digests and action caching.
pub mod sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    pub fn digest(data: &[u8]) -> [u8; 32] {
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];

        let bit_len = (data.len() as u64).wrapping_mul(8);
        let mut msg = data.to_vec();
        msg.push(0x80);
        while (msg.len() % 64) != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&bit_len.to_be_bytes());

        for chunk in msg.chunks_exact(64) {
            let mut w = [0u32; 64];
            for (i, word) in chunk.chunks_exact(4).enumerate() {
                w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }

            let mut a = h[0];
            let mut b = h[1];
            let mut c = h[2];
            let mut d = h[3];
            let mut e = h[4];
            let mut f = h[5];
            let mut g = h[6];
            let mut hh = h[7];

            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let temp1 = hh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(maj);

                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }

            h[0] = h[0].wrapping_add(a);
            h[1] = h[1].wrapping_add(b);
            h[2] = h[2].wrapping_add(c);
            h[3] = h[3].wrapping_add(d);
            h[4] = h[4].wrapping_add(e);
            h[5] = h[5].wrapping_add(f);
            h[6] = h[6].wrapping_add(g);
            h[7] = h[7].wrapping_add(hh);
        }

        let mut out = [0u8; 32];
        for (i, word) in h.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    pub fn hex_digest(data: &[u8]) -> String {
        super::to_hex(&digest(data))
    }
}

pub fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Incremental fingerprint builder matching `com.google.devtools.build.lib.util.Fingerprint`.
#[derive(Debug, Clone, Default)]
pub struct Fingerprint {
    buffer: Vec<u8>,
}

impl Fingerprint {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn add_bytes(&mut self, bytes: &[u8]) -> &mut Self {
        self.buffer
            .extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        self.buffer.extend_from_slice(bytes);
        self
    }

    pub fn add_str(&mut self, s: &str) -> &mut Self {
        self.add_bytes(s.as_bytes())
    }

    pub fn add_bool(&mut self, b: bool) -> &mut Self {
        self.buffer.push(u8::from(b));
        self
    }

    pub fn add_i32(&mut self, v: i32) -> &mut Self {
        self.buffer.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn add_u64(&mut self, v: u64) -> &mut Self {
        self.buffer.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn add_strings<I, S>(&mut self, iter: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let items: Vec<S> = iter.into_iter().collect();
        self.add_i32(items.len() as i32);
        for item in items {
            self.add_str(item.as_ref());
        }
        self
    }

    pub fn digest_and_reset(&mut self) -> [u8; 32] {
        let d = sha256::digest(&self.buffer);
        self.buffer.clear();
        d
    }

    pub fn hex_digest_and_reset(&mut self) -> String {
        to_hex(&self.digest_and_reset())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_md5_test_vectors() {
        assert_eq!(md5::hex_digest(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5::hex_digest(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            md5::hex_digest(b"The quick brown fox jumps over the lazy dog"),
            "9e107d9d372bb6826bd81d3542a419d6"
        );
    }

    #[test]
    fn test_sha256_test_vectors() {
        assert_eq!(
            sha256::hex_digest(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256::hex_digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}

