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

//! Bazel Event reporting (`Event`, `EventKind`, `Reporter`, `EventBus`, and Build Event Protocol).
//! Mirrors `src/main/java/com/google/devtools/build/lib/events` and `buildeventstream`.

use bazel_vfs::{FileSystem, PathFragment};
use std::fmt;
use std::io;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    Error,
    Warning,
    Info,
    Progress,
    Debug,
    Start,
    Finish,
    Subcommand,
    Stdout,
    Stderr,
    Pass,
    Fail,
}

impl EventKind {
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::Error => "ERROR: ",
            Self::Warning => "WARNING: ",
            Self::Info => "INFO: ",
            Self::Progress => "INFO: ",
            Self::Debug => "DEBUG: ",
            Self::Start => "INFO: ",
            Self::Finish => "INFO: ",
            Self::Subcommand => "SUBCOMMAND: ",
            Self::Stdout => "",
            Self::Stderr => "",
            Self::Pass => "PASS: ",
            Self::Fail => "FAIL: ",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    pub file: String,
    pub line: usize,
    pub column: usize,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub kind: EventKind,
    pub location: Option<Location>,
    pub message: String,
    pub tag: Option<String>,
}

impl Event {
    pub fn error(location: Option<Location>, message: impl Into<String>) -> Self {
        Self {
            kind: EventKind::Error,
            location,
            message: message.into(),
            tag: None,
        }
    }

    pub fn warning(location: Option<Location>, message: impl Into<String>) -> Self {
        Self {
            kind: EventKind::Warning,
            location,
            message: message.into(),
            tag: None,
        }
    }

    pub fn info(message: impl Into<String>) -> Self {
        Self {
            kind: EventKind::Info,
            location: None,
            message: message.into(),
            tag: None,
        }
    }

    pub fn subcommand(message: impl Into<String>) -> Self {
        Self {
            kind: EventKind::Subcommand,
            location: None,
            message: message.into(),
            tag: None,
        }
    }

    pub fn format_console_line(&self) -> String {
        match &self.location {
            Some(loc) => format!("{}{}: {}", self.kind.prefix(), loc, self.message),
            None => format!("{}{}", self.kind.prefix(), self.message),
        }
    }
}

/// Collects events and Build Event Protocol (BEP) notifications during a command execution.
#[derive(Debug, Default)]
pub struct Reporter {
    events: Mutex<Vec<Event>>,
    bep_events: Mutex<Vec<BepEvent>>,
}

impl Reporter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn handle(&self, event: Event) {
        self.events.lock().unwrap().push(event);
    }

    pub fn post_bep(&self, bep: BepEvent) {
        self.bep_events.lock().unwrap().push(bep);
    }

    pub fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }

    pub fn has_errors(&self) -> bool {
        self.events
            .lock()
            .unwrap()
            .iter()
            .any(|e| e.kind == EventKind::Error)
    }

    /// Writes Build Event Protocol (BEP) JSON lines to `--build_event_json_file`.
    pub fn write_bep_json_file(&self, fs: &dyn FileSystem, path: &PathFragment) -> io::Result<()> {
        let mut out = String::new();
        for ev in self.bep_events.lock().unwrap().iter() {
            out.push_str(&ev.to_json_line());
            out.push('\n');
        }
        fs.write_file(path, out.as_bytes())
    }
}

/// High-level Build Event Protocol (`build_event_stream.proto`) event variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BepEvent {
    Started {
        uuid: String,
        command: String,
        workspace_directory: String,
        server_pid: u32,
    },
    Progress {
        stdout: String,
        stderr: String,
    },
    TargetConfigured {
        label: String,
        rule_kind: String,
    },
    TargetComplete {
        label: String,
        success: bool,
    },
    TestResult {
        label: String,
        status: String,
        duration_ms: u64,
    },
    Finished {
        exit_code_name: String,
        exit_code: i32,
    },
}

impl BepEvent {
    pub fn to_json_line(&self) -> String {
        match self {
            Self::Started {
                uuid,
                command,
                workspace_directory,
                server_pid,
            } => format!(
                "{{\"id\":{{\"started\":{{}}}},\"started\":{{\"uuid\":\"{}\",\"command\":\"{}\",\"workspaceDirectory\":\"{}\",\"serverPid\":{}}}}}",
                escape_json(uuid),
                escape_json(command),
                escape_json(workspace_directory),
                server_pid
            ),
            Self::Progress { stdout, stderr } => format!(
                "{{\"id\":{{\"progress\":{{}}}},\"progress\":{{\"stdout\":\"{}\",\"stderr\":\"{}\"}}}}",
                escape_json(stdout),
                escape_json(stderr)
            ),
            Self::TargetConfigured { label, rule_kind } => format!(
                "{{\"id\":{{\"targetConfigured\":{{\"label\":\"{}\"}}}},\"configured\":{{\"targetKind\":\"{} rule\"}}}}",
                escape_json(label),
                escape_json(rule_kind)
            ),
            Self::TargetComplete { label, success } => format!(
                "{{\"id\":{{\"targetCompleted\":{{\"label\":\"{}\"}}}},\"completed\":{{\"success\":{}}}}}",
                escape_json(label),
                success
            ),
            Self::TestResult {
                label,
                status,
                duration_ms,
            } => format!(
                "{{\"id\":{{\"testResult\":{{\"label\":\"{}\"}}}},\"testResult\":{{\"status\":\"{}\",\"testAttemptDurationMillis\":\"{}\"}}}}",
                escape_json(label),
                escape_json(status),
                duration_ms
            ),
            Self::Finished {
                exit_code_name,
                exit_code,
            } => format!(
                "{{\"id\":{{\"buildFinished\":{{}}}},\"finished\":{{\"exitCode\":{{\"name\":\"{}\",\"code\":{}}}}}}}",
                escape_json(exit_code_name),
                exit_code
            ),
        }
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use bazel_vfs::InMemoryFileSystem;

    #[test]
    fn test_reporter_and_bep_json() {
        let reporter = Reporter::new();
        reporter.handle(Event::info("Loading package //src"));
        reporter.post_bep(BepEvent::Started {
            uuid: "uuid-1".to_string(),
            command: "build".to_string(),
            workspace_directory: "/workspace".to_string(),
            server_pid: 1234,
        });
        reporter.post_bep(BepEvent::Finished {
            exit_code_name: "SUCCESS".to_string(),
            exit_code: 0,
        });

        let fs = InMemoryFileSystem::default();
        let bep_path = PathFragment::create("/bep.json");
        reporter.write_bep_json_file(&fs, &bep_path).unwrap();
        let content = String::from_utf8(fs.read_file(&bep_path).unwrap()).unwrap();
        assert!(content.contains("\"command\":\"build\""));
        assert!(content.contains("\"name\":\"SUCCESS\""));
    }
}

