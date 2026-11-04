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

//! Protobuf wire-compatible types and gRPC framing for `command_server.proto`
//! and `failure_details.proto`.

use std::fmt;

/// Error returned when decoding a protobuf message or gRPC frame fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError(pub String);

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "protobuf decode error: {}", self.0)
    }
}

impl std::error::Error for DecodeError {}

/// Low-level Protobuf binary wire-format primitives (proto3).
pub mod wire {
    use super::DecodeError;

    pub const WIRE_TYPE_VARINT: u32 = 0;
    pub const WIRE_TYPE_64BIT: u32 = 1;
    pub const WIRE_TYPE_LENGTH_DELIMITED: u32 = 2;
    pub const WIRE_TYPE_32BIT: u32 = 5;

    pub fn encode_varint(mut val: u64, buf: &mut Vec<u8>) {
        loop {
            if (val & !0x7F) == 0 {
                buf.push(val as u8);
                return;
            } else {
                buf.push(((val & 0x7F) | 0x80) as u8);
                val >>= 7;
            }
        }
    }

    pub fn decode_varint(buf: &[u8], pos: &mut usize) -> Result<u64, DecodeError> {
        let mut result: u64 = 0;
        let mut shift = 0;
        while *pos < buf.len() {
            let b = buf[*pos];
            *pos += 1;
            result |= ((b & 0x7F) as u64) << shift;
            if (b & 0x80) == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err(DecodeError("varint overflow".to_string()));
            }
        }
        Err(DecodeError("unexpected EOF decoding varint".to_string()))
    }

    pub fn encode_tag(field_number: u32, wire_type: u32, buf: &mut Vec<u8>) {
        encode_varint(((field_number << 3) | wire_type) as u64, buf);
    }

    pub fn decode_tag(buf: &[u8], pos: &mut usize) -> Result<(u32, u32), DecodeError> {
        let v = decode_varint(buf, pos)? as u32;
        Ok((v >> 3, v & 0x7))
    }

    pub fn encode_bytes(field_number: u32, bytes: &[u8], buf: &mut Vec<u8>) {
        if !bytes.is_empty() {
            encode_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED, buf);
            encode_varint(bytes.len() as u64, buf);
            buf.extend_from_slice(bytes);
        }
    }

    pub fn encode_bytes_always(field_number: u32, bytes: &[u8], buf: &mut Vec<u8>) {
        encode_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED, buf);
        encode_varint(bytes.len() as u64, buf);
        buf.extend_from_slice(bytes);
    }

    pub fn encode_string(field_number: u32, s: &str, buf: &mut Vec<u8>) {
        encode_bytes(field_number, s.as_bytes(), buf);
    }

    pub fn encode_bool(field_number: u32, val: bool, buf: &mut Vec<u8>) {
        if val {
            encode_tag(field_number, WIRE_TYPE_VARINT, buf);
            buf.push(1);
        }
    }

    pub fn encode_int32(field_number: u32, val: i32, buf: &mut Vec<u8>) {
        if val != 0 {
            encode_tag(field_number, WIRE_TYPE_VARINT, buf);
            encode_varint(val as i64 as u64, buf);
        }
    }

    pub fn encode_int64(field_number: u32, val: i64, buf: &mut Vec<u8>) {
        if val != 0 {
            encode_tag(field_number, WIRE_TYPE_VARINT, buf);
            encode_varint(val as u64, buf);
        }
    }

    pub fn decode_bytes<'a>(buf: &'a [u8], pos: &mut usize) -> Result<&'a [u8], DecodeError> {
        let len = decode_varint(buf, pos)? as usize;
        if *pos + len > buf.len() {
            return Err(DecodeError(
                "unexpected EOF decoding length-delimited field".to_string(),
            ));
        }
        let slice = &buf[*pos..*pos + len];
        *pos += len;
        Ok(slice)
    }

    pub fn decode_string(buf: &[u8], pos: &mut usize) -> Result<String, DecodeError> {
        let bytes = decode_bytes(buf, pos)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| DecodeError(format!("invalid UTF-8 in string field: {e}")))
    }

    pub fn skip_field(wire_type: u32, buf: &[u8], pos: &mut usize) -> Result<(), DecodeError> {
        match wire_type {
            WIRE_TYPE_VARINT => {
                let _ = decode_varint(buf, pos)?;
                Ok(())
            }
            WIRE_TYPE_64BIT => {
                if *pos + 8 > buf.len() {
                    return Err(DecodeError("unexpected EOF skipping 64-bit".to_string()));
                }
                *pos += 8;
                Ok(())
            }
            WIRE_TYPE_LENGTH_DELIMITED => {
                let _ = decode_bytes(buf, pos)?;
                Ok(())
            }
            WIRE_TYPE_32BIT => {
                if *pos + 4 > buf.len() {
                    return Err(DecodeError("unexpected EOF skipping 32-bit".to_string()));
                }
                *pos += 4;
                Ok(())
            }
            other => Err(DecodeError(format!("unsupported wire type: {other}"))),
        }
    }
}

/// `google.protobuf.Any`
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Any {
    pub type_url: String,
    pub value: Vec<u8>,
}

impl Any {
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        wire::encode_string(1, &self.type_url, &mut buf);
        wire::encode_bytes(2, &self.value, &mut buf);
        buf
    }

    pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
        let mut msg = Self::default();
        let mut pos = 0;
        while pos < buf.len() {
            let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
            match (field_num, wire_type) {
                (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                    msg.type_url = wire::decode_string(buf, &mut pos)?;
                }
                (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                    msg.value = wire::decode_bytes(buf, &mut pos)?.to_vec();
                }
                _ => wire::skip_field(wire_type, buf, &mut pos)?,
            }
        }
        Ok(msg)
    }
}

/// Types from `src/main/protobuf/failure_details.proto`.
pub mod failure_details {
    use super::{wire, DecodeError};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    #[repr(i32)]
    pub enum InterruptedCode {
        #[default]
        InterruptedUnknown = 0,
        Interrupted = 4,
        Build = 5,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    #[repr(i32)]
    pub enum CommandCode {
        #[default]
        CommandFailureUnknown = 0,
        CommandNotFound = 1,
        AnotherCommandRunning = 2,
        PreviouslyShutdown = 3,
        StarlarkCpuProfileFileInitializationFailure = 4,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    #[repr(i32)]
    pub enum GrpcServerCode {
        #[default]
        GrpcServerUnknown = 0,
        NotCompiledIn = 1,
        ServerBindFailure = 2,
        BadCookie = 3,
        NoClientDescription = 4,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Category {
        Interrupted(InterruptedCode),
        Command(CommandCode),
        GrpcServer(GrpcServerCode),
        Submessage { field_number: u32, code: i32 },
    }

    impl Category {
        pub fn field_number(&self) -> u32 {
            match self {
                Category::Interrupted(_) => 101,
                Category::Command(_) => 122,
                Category::GrpcServer(_) => 124,
                Category::Submessage { field_number, .. } => *field_number,
            }
        }

        pub fn subcategory_code(&self) -> i32 {
            match self {
                Category::Interrupted(c) => *c as i32,
                Category::Command(c) => *c as i32,
                Category::GrpcServer(c) => *c as i32,
                Category::Submessage { code, .. } => *code,
            }
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct FailureDetail {
        pub message: String,
        pub category: Option<Category>,
    }

    impl FailureDetail {
        pub fn new(message: impl Into<String>, category: Category) -> Self {
            Self {
                message: message.into(),
                category: Some(category),
            }
        }

        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.message, &mut buf);
            if let Some(ref cat) = self.category {
                let mut sub = Vec::new();
                wire::encode_int32(1, cat.subcategory_code(), &mut sub);
                wire::encode_bytes_always(cat.field_number(), &sub, &mut buf);
            }
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut detail = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        detail.message = wire::decode_string(buf, &mut pos)?;
                    }
                    (f @ 101..=10000, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub_bytes = wire::decode_bytes(buf, &mut pos)?;
                        let mut sub_pos = 0;
                        let mut code = 0i32;
                        while sub_pos < sub_bytes.len() {
                            let (sf, sw) = wire::decode_tag(sub_bytes, &mut sub_pos)?;
                            if sf == 1 && sw == wire::WIRE_TYPE_VARINT {
                                code = wire::decode_varint(sub_bytes, &mut sub_pos)? as i32;
                            } else {
                                wire::skip_field(sw, sub_bytes, &mut sub_pos)?;
                            }
                        }
                        detail.category = Some(match f {
                            101 => Category::Interrupted(match code {
                                4 => InterruptedCode::Interrupted,
                                5 => InterruptedCode::Build,
                                _ => InterruptedCode::InterruptedUnknown,
                            }),
                            122 => Category::Command(match code {
                                1 => CommandCode::CommandNotFound,
                                2 => CommandCode::AnotherCommandRunning,
                                3 => CommandCode::PreviouslyShutdown,
                                4 => CommandCode::StarlarkCpuProfileFileInitializationFailure,
                                _ => CommandCode::CommandFailureUnknown,
                            }),
                            124 => Category::GrpcServer(match code {
                                1 => GrpcServerCode::NotCompiledIn,
                                2 => GrpcServerCode::ServerBindFailure,
                                3 => GrpcServerCode::BadCookie,
                                4 => GrpcServerCode::NoClientDescription,
                                _ => GrpcServerCode::GrpcServerUnknown,
                            }),
                            _ => Category::Submessage {
                                field_number: f,
                                code,
                            },
                        });
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(detail)
        }
    }
}

/// Types from `src/main/protobuf/command_server.proto`.
pub mod command_server {
    use super::failure_details::FailureDetail;
    use super::{wire, Any, DecodeError};

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct StartupOption {
        pub option: Vec<u8>,
        pub source: Vec<u8>,
    }

    impl StartupOption {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_bytes(1, &self.option, &mut buf);
            wire::encode_bytes(2, &self.source, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.option = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.source = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct EnvironmentVariable {
        pub name: Vec<u8>,
        pub value: Vec<u8>,
    }

    impl EnvironmentVariable {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_bytes(1, &self.name, &mut buf);
            wire::encode_bytes(2, &self.value, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.name = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.value = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct ScriptPath {
        pub script_path: Vec<u8>,
        pub script_contents: Vec<u8>,
    }

    impl ScriptPath {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_bytes(1, &self.script_path, &mut buf);
            wire::encode_bytes(2, &self.script_contents, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.script_path = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.script_contents = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    #[repr(i32)]
    pub enum PathToReplaceType {
        #[default]
        Unspecified = 0,
        OutputBase = 1,
        BuildWorkingDirectory = 2,
        BuildWorkspaceDirectory = 3,
        TestLogSubdir = 4,
        Home = 5,
        TempLoggingDirectory = 7,
    }

    impl PathToReplaceType {
        pub fn from_i32(v: i32) -> Self {
            match v {
                1 => Self::OutputBase,
                2 => Self::BuildWorkingDirectory,
                3 => Self::BuildWorkspaceDirectory,
                4 => Self::TestLogSubdir,
                5 => Self::Home,
                7 => Self::TempLoggingDirectory,
                _ => Self::Unspecified,
            }
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct PathToReplace {
        pub r#type: PathToReplaceType,
        pub value: Vec<u8>,
    }

    impl PathToReplace {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_int32(1, self.r#type as i32, &mut buf);
            wire::encode_bytes(2, &self.value, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_VARINT) => {
                        msg.r#type = PathToReplaceType::from_i32(
                            wire::decode_varint(buf, &mut pos)? as i32,
                        );
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.value = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct ExecRequest {
        pub working_directory: Vec<u8>,
        pub argv: Vec<Vec<u8>>,
        pub environment_variable: Vec<EnvironmentVariable>,
        pub environment_variable_to_clear: Vec<Vec<u8>>,
        pub should_exec: bool,
        pub script_path: Option<ScriptPath>,
        pub path_to_replace: Vec<PathToReplace>,
    }

    impl ExecRequest {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_bytes(1, &self.working_directory, &mut buf);
            for arg in &self.argv {
                wire::encode_bytes_always(2, arg, &mut buf);
            }
            for ev in &self.environment_variable {
                wire::encode_bytes_always(3, &ev.encode_to_vec(), &mut buf);
            }
            for clear in &self.environment_variable_to_clear {
                wire::encode_bytes_always(4, clear, &mut buf);
            }
            wire::encode_bool(5, self.should_exec, &mut buf);
            if let Some(ref sp) = self.script_path {
                wire::encode_bytes_always(6, &sp.encode_to_vec(), &mut buf);
            }
            for ptr in &self.path_to_replace {
                wire::encode_bytes_always(7, &ptr.encode_to_vec(), &mut buf);
            }
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.working_directory = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.argv.push(wire::decode_bytes(buf, &mut pos)?.to_vec());
                    }
                    (3, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.environment_variable
                            .push(EnvironmentVariable::decode(sub)?);
                    }
                    (4, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.environment_variable_to_clear
                            .push(wire::decode_bytes(buf, &mut pos)?.to_vec());
                    }
                    (5, wire::WIRE_TYPE_VARINT) => {
                        msg.should_exec = wire::decode_varint(buf, &mut pos)? != 0;
                    }
                    (6, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.script_path = Some(ScriptPath::decode(sub)?);
                    }
                    (7, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.path_to_replace.push(PathToReplace::decode(sub)?);
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct RunRequest {
        pub cookie: String,
        pub arg: Vec<Vec<u8>>,
        pub block_for_lock: bool,
        pub client_description: String,
        pub invocation_policy: String,
        pub startup_options: Vec<StartupOption>,
        pub preemptible: bool,
        pub command_extensions: Vec<Any>,
        pub quiet: bool,
        pub block_for_lock_timeout_ms: Option<i64>,
    }

    impl RunRequest {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.cookie, &mut buf);
            for a in &self.arg {
                wire::encode_bytes_always(2, a, &mut buf);
            }
            wire::encode_bool(3, self.block_for_lock, &mut buf);
            wire::encode_string(4, &self.client_description, &mut buf);
            wire::encode_string(5, &self.invocation_policy, &mut buf);
            for so in &self.startup_options {
                wire::encode_bytes_always(6, &so.encode_to_vec(), &mut buf);
            }
            wire::encode_bool(7, self.preemptible, &mut buf);
            for ext in &self.command_extensions {
                wire::encode_bytes_always(8, &ext.encode_to_vec(), &mut buf);
            }
            wire::encode_bool(9, self.quiet, &mut buf);
            if let Some(timeout) = self.block_for_lock_timeout_ms {
                wire::encode_tag(10, wire::WIRE_TYPE_VARINT, &mut buf);
                wire::encode_varint(timeout as u64, &mut buf);
            }
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.cookie = wire::decode_string(buf, &mut pos)?;
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.arg.push(wire::decode_bytes(buf, &mut pos)?.to_vec());
                    }
                    (3, wire::WIRE_TYPE_VARINT) => {
                        msg.block_for_lock = wire::decode_varint(buf, &mut pos)? != 0;
                    }
                    (4, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.client_description = wire::decode_string(buf, &mut pos)?;
                    }
                    (5, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.invocation_policy = wire::decode_string(buf, &mut pos)?;
                    }
                    (6, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.startup_options.push(StartupOption::decode(sub)?);
                    }
                    (7, wire::WIRE_TYPE_VARINT) => {
                        msg.preemptible = wire::decode_varint(buf, &mut pos)? != 0;
                    }
                    (8, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.command_extensions.push(Any::decode(sub)?);
                    }
                    (9, wire::WIRE_TYPE_VARINT) => {
                        msg.quiet = wire::decode_varint(buf, &mut pos)? != 0;
                    }
                    (10, wire::WIRE_TYPE_VARINT) => {
                        msg.block_for_lock_timeout_ms =
                            Some(wire::decode_varint(buf, &mut pos)? as i64);
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct RunResponse {
        pub cookie: String,
        pub standard_output: Vec<u8>,
        pub standard_error: Vec<u8>,
        pub finished: bool,
        pub exit_code: i32,
        pub command_id: String,
        pub termination_expected: bool,
        pub exec_request: Option<ExecRequest>,
        pub failure_detail: Option<FailureDetail>,
        pub command_extensions: Vec<Any>,
    }

    impl RunResponse {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.cookie, &mut buf);
            wire::encode_bytes(2, &self.standard_output, &mut buf);
            wire::encode_bytes(3, &self.standard_error, &mut buf);
            wire::encode_bool(4, self.finished, &mut buf);
            wire::encode_int32(5, self.exit_code, &mut buf);
            wire::encode_string(6, &self.command_id, &mut buf);
            wire::encode_bool(7, self.termination_expected, &mut buf);
            if let Some(ref er) = self.exec_request {
                wire::encode_bytes_always(8, &er.encode_to_vec(), &mut buf);
            }
            if let Some(ref fd) = self.failure_detail {
                wire::encode_bytes_always(9, &fd.encode_to_vec(), &mut buf);
            }
            for ext in &self.command_extensions {
                wire::encode_bytes_always(10, &ext.encode_to_vec(), &mut buf);
            }
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.cookie = wire::decode_string(buf, &mut pos)?;
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.standard_output = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    (3, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.standard_error = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    (4, wire::WIRE_TYPE_VARINT) => {
                        msg.finished = wire::decode_varint(buf, &mut pos)? != 0;
                    }
                    (5, wire::WIRE_TYPE_VARINT) => {
                        msg.exit_code = wire::decode_varint(buf, &mut pos)? as i32;
                    }
                    (6, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.command_id = wire::decode_string(buf, &mut pos)?;
                    }
                    (7, wire::WIRE_TYPE_VARINT) => {
                        msg.termination_expected = wire::decode_varint(buf, &mut pos)? != 0;
                    }
                    (8, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.exec_request = Some(ExecRequest::decode(sub)?);
                    }
                    (9, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.failure_detail = Some(FailureDetail::decode(sub)?);
                    }
                    (10, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.command_extensions.push(Any::decode(sub)?);
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct CancelRequest {
        pub cookie: String,
        pub command_id: String,
    }

    impl CancelRequest {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.cookie, &mut buf);
            wire::encode_string(2, &self.command_id, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.cookie = wire::decode_string(buf, &mut pos)?;
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.command_id = wire::decode_string(buf, &mut pos)?;
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct CancelResponse {
        pub cookie: String,
    }

    impl CancelResponse {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.cookie, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.cookie = wire::decode_string(buf, &mut pos)?;
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct TerminalSizeRequest {
        pub cookie: String,
        pub command_id: String,
        pub columns: i32,
        pub rows: i32,
    }

    impl TerminalSizeRequest {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.cookie, &mut buf);
            wire::encode_string(2, &self.command_id, &mut buf);
            wire::encode_int32(3, self.columns, &mut buf);
            wire::encode_int32(4, self.rows, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.cookie = wire::decode_string(buf, &mut pos)?;
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.command_id = wire::decode_string(buf, &mut pos)?;
                    }
                    (3, wire::WIRE_TYPE_VARINT) => {
                        msg.columns = wire::decode_varint(buf, &mut pos)? as i32;
                    }
                    (4, wire::WIRE_TYPE_VARINT) => {
                        msg.rows = wire::decode_varint(buf, &mut pos)? as i32;
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct TerminalSizeResponse {
        pub cookie: String,
    }

    impl TerminalSizeResponse {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.cookie, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.cookie = wire::decode_string(buf, &mut pos)?;
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct PingRequest {
        pub cookie: String,
    }

    impl PingRequest {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.cookie, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.cookie = wire::decode_string(buf, &mut pos)?;
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct PingResponse {
        pub cookie: String,
    }

    impl PingResponse {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.cookie, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.cookie = wire::decode_string(buf, &mut pos)?;
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct ServerInfo {
        pub pid: i32,
        pub address: String,
        pub request_cookie: String,
        pub response_cookie: String,
    }

    impl ServerInfo {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_int32(1, self.pid, &mut buf);
            wire::encode_string(2, &self.address, &mut buf);
            wire::encode_string(3, &self.request_cookie, &mut buf);
            wire::encode_string(4, &self.response_cookie, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_VARINT) => {
                        msg.pid = wire::decode_varint(buf, &mut pos)? as i32;
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.address = wire::decode_string(buf, &mut pos)?;
                    }
                    (3, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.request_cookie = wire::decode_string(buf, &mut pos)?;
                    }
                    (4, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.response_cookie = wire::decode_string(buf, &mut pos)?;
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct InfoItem {
        pub key: String,
        pub value: Vec<u8>,
    }

    impl InfoItem {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.key, &mut buf);
            wire::encode_bytes(2, &self.value, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.key = wire::decode_string(buf, &mut pos)?;
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.value = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct InfoResponse {
        pub path_to_replace: Vec<PathToReplace>,
        pub info_item: Vec<InfoItem>,
        pub print_keys: bool,
    }

    impl InfoResponse {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            for ptr in &self.path_to_replace {
                wire::encode_bytes_always(1, &ptr.encode_to_vec(), &mut buf);
            }
            for item in &self.info_item {
                wire::encode_bytes_always(2, &item.encode_to_vec(), &mut buf);
            }
            wire::encode_bool(3, self.print_keys, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (field_num, wire_type) = wire::decode_tag(buf, &mut pos)?;
                match (field_num, wire_type) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.path_to_replace.push(PathToReplace::decode(sub)?);
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.info_item.push(InfoItem::decode(sub)?);
                    }
                    (3, wire::WIRE_TYPE_VARINT) => {
                        msg.print_keys = wire::decode_varint(buf, &mut pos)? != 0;
                    }
                    _ => wire::skip_field(wire_type, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }
}

/// Standard 5-byte gRPC length-prefixed framing (`[compression_flag: u8][length: u32 BE][payload]`).
pub mod grpc_framing {
    use std::io::{self, Read, Write};

    pub fn write_grpc_frame<W: Write>(writer: &mut W, payload: &[u8]) -> io::Result<()> {
        let mut header = [0u8; 5];
        header[0] = 0; // uncompressed
        let len = payload.len() as u32;
        header[1..5].copy_from_slice(&len.to_be_bytes());
        writer.write_all(&header)?;
        writer.write_all(payload)?;
        writer.flush()
    }

    pub fn read_grpc_frame<R: Read>(reader: &mut R) -> io::Result<Option<Vec<u8>>> {
        let mut header = [0u8; 5];
        match reader.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
        let len = u32::from_be_bytes([header[1], header[2], header[3], header[4]]) as usize;
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        Ok(Some(buf))
    }
}

/// Persistent worker protocol messages from `src/main/protobuf/worker_protocol.proto`.
pub mod worker_protocol {
    use super::{wire, DecodeError};

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct Input {
        pub path: String,
        pub digest: Vec<u8>,
    }

    impl Input {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.path, &mut buf);
            wire::encode_bytes(2, &self.digest, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (f, w) = wire::decode_tag(buf, &mut pos)?;
                match (f, w) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.path = wire::decode_string(buf, &mut pos)?;
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.digest = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    _ => wire::skip_field(w, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct WorkRequest {
        pub arguments: Vec<String>,
        pub inputs: Vec<Input>,
        pub request_id: i32,
        pub cancel: bool,
        pub verbosity: i32,
        pub sandbox_dir: String,
    }

    impl WorkRequest {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            for arg in &self.arguments {
                wire::encode_string(1, arg, &mut buf);
            }
            for inp in &self.inputs {
                wire::encode_bytes_always(2, &inp.encode_to_vec(), &mut buf);
            }
            wire::encode_int32(3, self.request_id, &mut buf);
            wire::encode_bool(4, self.cancel, &mut buf);
            wire::encode_int32(5, self.verbosity, &mut buf);
            wire::encode_string(6, &self.sandbox_dir, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (f, w) = wire::decode_tag(buf, &mut pos)?;
                match (f, w) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.arguments.push(wire::decode_string(buf, &mut pos)?);
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.inputs.push(Input::decode(sub)?);
                    }
                    (3, wire::WIRE_TYPE_VARINT) => {
                        msg.request_id = wire::decode_varint(buf, &mut pos)? as i32;
                    }
                    (4, wire::WIRE_TYPE_VARINT) => {
                        msg.cancel = wire::decode_varint(buf, &mut pos)? != 0;
                    }
                    (5, wire::WIRE_TYPE_VARINT) => {
                        msg.verbosity = wire::decode_varint(buf, &mut pos)? as i32;
                    }
                    (6, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.sandbox_dir = wire::decode_string(buf, &mut pos)?;
                    }
                    _ => wire::skip_field(w, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct WorkResponse {
        pub exit_code: i32,
        pub output: String,
        pub request_id: i32,
        pub was_cancelled: bool,
    }

    impl WorkResponse {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_int32(1, self.exit_code, &mut buf);
            wire::encode_string(2, &self.output, &mut buf);
            wire::encode_int32(3, self.request_id, &mut buf);
            wire::encode_bool(4, self.was_cancelled, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (f, w) = wire::decode_tag(buf, &mut pos)?;
                match (f, w) {
                    (1, wire::WIRE_TYPE_VARINT) => {
                        msg.exit_code = wire::decode_varint(buf, &mut pos)? as i32;
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.output = wire::decode_string(buf, &mut pos)?;
                    }
                    (3, wire::WIRE_TYPE_VARINT) => {
                        msg.request_id = wire::decode_varint(buf, &mut pos)? as i32;
                    }
                    (4, wire::WIRE_TYPE_VARINT) => {
                        msg.was_cancelled = wire::decode_varint(buf, &mut pos)? != 0;
                    }
                    _ => wire::skip_field(w, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }
}

/// Remote Execution API v2 (`build.bazel.remote.execution.v2`) core messages.
pub mod remote_execution {
    use super::{wire, DecodeError};

    #[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Digest {
        pub hash: String,
        pub size_bytes: i64,
    }

    impl Digest {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.hash, &mut buf);
            wire::encode_int64(2, self.size_bytes, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (f, w) = wire::decode_tag(buf, &mut pos)?;
                match (f, w) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.hash = wire::decode_string(buf, &mut pos)?;
                    }
                    (2, wire::WIRE_TYPE_VARINT) => {
                        msg.size_bytes = wire::decode_varint(buf, &mut pos)? as i64;
                    }
                    _ => wire::skip_field(w, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct CommandEnvironmentVariable {
        pub name: String,
        pub value: String,
    }

    impl CommandEnvironmentVariable {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.name, &mut buf);
            wire::encode_string(2, &self.value, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (f, w) = wire::decode_tag(buf, &mut pos)?;
                match (f, w) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.name = wire::decode_string(buf, &mut pos)?;
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.value = wire::decode_string(buf, &mut pos)?;
                    }
                    _ => wire::skip_field(w, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct Command {
        pub arguments: Vec<String>,
        pub environment_variables: Vec<CommandEnvironmentVariable>,
        pub output_files: Vec<String>,
        pub output_directories: Vec<String>,
        pub working_directory: String,
    }

    impl Command {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            for a in &self.arguments {
                wire::encode_string(1, a, &mut buf);
            }
            for ev in &self.environment_variables {
                wire::encode_bytes_always(2, &ev.encode_to_vec(), &mut buf);
            }
            for of in &self.output_files {
                wire::encode_string(3, of, &mut buf);
            }
            for od in &self.output_directories {
                wire::encode_string(4, od, &mut buf);
            }
            wire::encode_string(6, &self.working_directory, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (f, w) = wire::decode_tag(buf, &mut pos)?;
                match (f, w) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.arguments.push(wire::decode_string(buf, &mut pos)?);
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.environment_variables
                            .push(CommandEnvironmentVariable::decode(sub)?);
                    }
                    (3, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.output_files.push(wire::decode_string(buf, &mut pos)?);
                    }
                    (4, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.output_directories
                            .push(wire::decode_string(buf, &mut pos)?);
                    }
                    (6, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.working_directory = wire::decode_string(buf, &mut pos)?;
                    }
                    _ => wire::skip_field(w, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct Action {
        pub command_digest: Option<Digest>,
        pub input_root_digest: Option<Digest>,
        pub do_not_cache: bool,
    }

    impl Action {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            if let Some(ref cd) = self.command_digest {
                wire::encode_bytes_always(1, &cd.encode_to_vec(), &mut buf);
            }
            if let Some(ref ir) = self.input_root_digest {
                wire::encode_bytes_always(2, &ir.encode_to_vec(), &mut buf);
            }
            wire::encode_bool(6, self.do_not_cache, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (f, w) = wire::decode_tag(buf, &mut pos)?;
                match (f, w) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.command_digest = Some(Digest::decode(sub)?);
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.input_root_digest = Some(Digest::decode(sub)?);
                    }
                    (6, wire::WIRE_TYPE_VARINT) => {
                        msg.do_not_cache = wire::decode_varint(buf, &mut pos)? != 0;
                    }
                    _ => wire::skip_field(w, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct OutputFile {
        pub path: String,
        pub digest: Option<Digest>,
        pub is_executable: bool,
    }

    impl OutputFile {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            wire::encode_string(1, &self.path, &mut buf);
            if let Some(ref d) = self.digest {
                wire::encode_bytes_always(2, &d.encode_to_vec(), &mut buf);
            }
            wire::encode_bool(4, self.is_executable, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (f, w) = wire::decode_tag(buf, &mut pos)?;
                match (f, w) {
                    (1, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.path = wire::decode_string(buf, &mut pos)?;
                    }
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.digest = Some(Digest::decode(sub)?);
                    }
                    (4, wire::WIRE_TYPE_VARINT) => {
                        msg.is_executable = wire::decode_varint(buf, &mut pos)? != 0;
                    }
                    _ => wire::skip_field(w, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct ActionResult {
        pub output_files: Vec<OutputFile>,
        pub exit_code: i32,
        pub stdout_raw: Vec<u8>,
        pub stderr_raw: Vec<u8>,
    }

    impl ActionResult {
        pub fn encode_to_vec(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            for of in &self.output_files {
                wire::encode_bytes_always(2, &of.encode_to_vec(), &mut buf);
            }
            wire::encode_int32(4, self.exit_code, &mut buf);
            wire::encode_bytes(5, &self.stdout_raw, &mut buf);
            wire::encode_bytes(7, &self.stderr_raw, &mut buf);
            buf
        }

        pub fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
            let mut msg = Self::default();
            let mut pos = 0;
            while pos < buf.len() {
                let (f, w) = wire::decode_tag(buf, &mut pos)?;
                match (f, w) {
                    (2, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        let sub = wire::decode_bytes(buf, &mut pos)?;
                        msg.output_files.push(OutputFile::decode(sub)?);
                    }
                    (4, wire::WIRE_TYPE_VARINT) => {
                        msg.exit_code = wire::decode_varint(buf, &mut pos)? as i32;
                    }
                    (5, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.stdout_raw = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    (7, wire::WIRE_TYPE_LENGTH_DELIMITED) => {
                        msg.stderr_raw = wire::decode_bytes(buf, &mut pos)?.to_vec();
                    }
                    _ => wire::skip_field(w, buf, &mut pos)?,
                }
            }
            Ok(msg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::command_server::*;
    use super::failure_details::*;

    #[test]
    fn test_server_info_roundtrip() {
        let info = ServerInfo {
            pid: 4242,
            address: "[::1]:51234".to_string(),
            request_cookie: "req_abc123".to_string(),
            response_cookie: "resp_def456".to_string(),
        };
        let bytes = info.encode_to_vec();
        let decoded = ServerInfo::decode(&bytes).unwrap();
        assert_eq!(info, decoded);
    }

    #[test]
    fn test_run_request_and_response_roundtrip() {
        let req = RunRequest {
            cookie: "cookie123".to_string(),
            arg: vec![b"build".to_vec(), b"//src:bazel".to_vec()],
            block_for_lock: true,
            client_description: "pid=1234".to_string(),
            invocation_policy: String::new(),
            startup_options: vec![StartupOption {
                option: b"--batch".to_vec(),
                source: b"".to_vec(),
            }],
            preemptible: false,
            command_extensions: vec![],
            quiet: false,
            block_for_lock_timeout_ms: Some(5000),
        };
        let encoded_req = req.encode_to_vec();
        let decoded_req = RunRequest::decode(&encoded_req).unwrap();
        assert_eq!(req, decoded_req);

        let resp = RunResponse {
            cookie: "resp_cookie".to_string(),
            standard_output: b"INFO: Build completed successfully\n".to_vec(),
            standard_error: Vec::new(),
            finished: true,
            exit_code: 0,
            command_id: "cmd-uuid-1".to_string(),
            termination_expected: false,
            exec_request: None,
            failure_detail: Some(FailureDetail::new(
                "all good",
                Category::Command(CommandCode::CommandNotFound),
            )),
            command_extensions: vec![],
        };
        let encoded_resp = resp.encode_to_vec();
        let decoded_resp = RunResponse::decode(&encoded_resp).unwrap();
        assert_eq!(resp, decoded_resp);
    }
}

