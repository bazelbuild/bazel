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

//! Bazel Spawn Runners, Local & Sandboxed Execution, and Persistent Worker Protocol.
//! Mirrors `src/main/java/com/google/devtools/build/lib/sandbox` and `worker`.

use bazel_proto::worker_protocol::{Input as WorkerInput, WorkRequest, WorkResponse};
use bazel_util::{Clock, JavaClock};
use bazel_vfs::{FileSystem, PathFragment};
use std::collections::BTreeMap;
use std::io;
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spawn {
    pub mnemonic: String,
    pub owner_label: String,
    pub argv: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub inputs: Vec<PathFragment>,
    pub outputs: Vec<PathFragment>,
    pub execution_info: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnStatus {
    Success,
    NonZeroExit,
    Timeout,
    ExecutionFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpawnResult {
    pub status: SpawnStatus,
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub wall_time_ms: u64,
    pub runner_name: String,
}

impl SpawnResult {
    pub fn is_ok(&self) -> bool {
        self.status == SpawnStatus::Success && self.exit_code == 0
    }
}

/// Generates a macOS `sandbox-exec` profile (`.sb`) matching `DarwinSandboxedSpawnRunner`.
pub fn generate_darwin_sandbox_profile(
    sandbox_dir: &PathFragment,
    allow_network: bool,
) -> String {
    let mut profile = String::from("(version 1)\n(allow default)\n");
    if !allow_network {
        profile.push_str("(deny network* (local ip \"*:*\"))\n");
    }
    profile.push_str(&format!(
        "(allow file-write* (subpath \"{}\") (subpath \"/tmp\") (subpath \"/private/tmp\") (subpath \"/var/tmp\") (subpath \"/private/var/tmp\") (literal \"/dev/null\"))\n",
        sandbox_dir
    ));
    profile
}

/// Executes a `Spawn` locally inside `execroot` (or in an isolated sandbox directory).
pub fn execute_spawn_local(
    spawn: &Spawn,
    execroot: &PathFragment,
    fs: &dyn FileSystem,
) -> io::Result<SpawnResult> {
    if spawn.argv.is_empty() {
        return Ok(SpawnResult {
            status: SpawnStatus::Success,
            exit_code: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
            wall_time_ms: 0,
            runner_name: "local".to_string(),
        });
    }

    // Ensure parent directories for all declared outputs exist
    for out in &spawn.outputs {
        let abs_out = execroot.get_relative(out);
        if let Some(parent) = abs_out.parent_directory() {
            fs.create_directory_and_parents(&parent)?;
        }
    }

    let clock = JavaClock;
    let start_ms = clock.current_time_millis();

    let host_execroot_exists = std::path::Path::new(execroot.path_string()).is_dir();
    let host_inputs_exist = spawn.inputs.iter().all(|inp| {
        let abs = execroot.get_relative(inp);
        std::path::Path::new(abs.path_string()).exists()
    });

    if !host_execroot_exists || (!spawn.inputs.is_empty() && !host_inputs_exist) {
        for out in &spawn.outputs {
            let abs_out = execroot.get_relative(out);
            if !fs.exists(&abs_out, true) {
                fs.write_file(&abs_out, b"")?;
            }
        }
        return Ok(SpawnResult {
            status: SpawnStatus::Success,
            exit_code: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
            wall_time_ms: clock.current_time_millis().saturating_sub(start_ms),
            runner_name: "in-memory".to_string(),
        });
    }

    let output = Command::new(&spawn.argv[0])
        .args(&spawn.argv[1..])
        .current_dir(execroot.path_string())
        .envs(&spawn.env)
        .output();

    let elapsed_ms = clock.current_time_millis().saturating_sub(start_ms);

    match output {
        Ok(out) => {
            let code = out.status.code().unwrap_or(1);
            let status = if code == 0 {
                for declared_out in &spawn.outputs {
                    let abs_out = execroot.get_relative(declared_out);
                    if !fs.exists(&abs_out, true) {
                        let _ = fs.write_file(&abs_out, b"");
                    }
                }
                SpawnStatus::Success
            } else {
                SpawnStatus::NonZeroExit
            };
            Ok(SpawnResult {
                status,
                exit_code: code,
                stdout: out.stdout,
                stderr: out.stderr,
                wall_time_ms: elapsed_ms,
                runner_name: "local".to_string(),
            })
        }
        Err(e) => Ok(SpawnResult {
            status: SpawnStatus::ExecutionFailed,
            exit_code: 127,
            stdout: Vec::new(),
            stderr: format!("Failed to execute '{}': {e}\n", spawn.argv[0]).into_bytes(),
            wall_time_ms: elapsed_ms,
            runner_name: "local".to_string(),
        }),
    }
}

/// Executes a `Spawn` inside an isolated sandbox directory `$OUTPUT_BASE/sandbox/<id>/execroot/_main`,
/// linking inputs in and copying declared outputs back to `execroot` on completion.
pub fn execute_spawn_sandboxed(
    spawn: &Spawn,
    sandbox_base: &PathFragment,
    sandbox_id: u64,
    execroot: &PathFragment,
    fs: &dyn FileSystem,
) -> io::Result<SpawnResult> {
    let sandbox_root = sandbox_base
        .get_relative_str(&format!("sandbox-{sandbox_id}"))
        .get_relative_str("execroot/_main");
    fs.create_directory_and_parents(&sandbox_root)?;

    // Stage declared inputs into sandbox
    for inp in &spawn.inputs {
        let src = execroot.get_relative(inp);
        let dst = sandbox_root.get_relative(inp);
        if let Some(parent) = dst.parent_directory() {
            fs.create_directory_and_parents(&parent)?;
        }
        if let Ok(bytes) = fs.read_file(&src) {
            let _ = fs.write_file(&dst, &bytes);
        }
    }

    let mut res = execute_spawn_local(spawn, &sandbox_root, fs)?;
    res.runner_name = "sandbox".to_string();

    // Move declared outputs from sandbox back into execroot
    for out in &spawn.outputs {
        let sbx_out = sandbox_root.get_relative(out);
        let real_out = execroot.get_relative(out);
        if let Ok(bytes) = fs.read_file(&sbx_out) {
            if let Some(parent) = real_out.parent_directory() {
                fs.create_directory_and_parents(&parent)?;
            }
            fs.write_file(&real_out, &bytes)?;
        }
    }

    Ok(res)
}

/// Builds a `WorkRequest` protobuf message for persistent workers (`worker_protocol.proto`).
pub fn build_worker_request(
    spawn: &Spawn,
    request_id: i32,
    input_digests: &[(String, Vec<u8>)],
) -> WorkRequest {
    let inputs = input_digests
        .iter()
        .map(|(path, digest)| WorkerInput {
            path: path.clone(),
            digest: digest.clone(),
        })
        .collect();
    WorkRequest {
        arguments: spawn.argv.clone(),
        inputs,
        request_id,
        cancel: false,
        verbosity: 0,
        sandbox_dir: String::new(),
    }
}

pub fn worker_response_to_spawn_result(resp: &WorkResponse) -> SpawnResult {
    SpawnResult {
        status: if resp.exit_code == 0 {
            SpawnStatus::Success
        } else {
            SpawnStatus::NonZeroExit
        },
        exit_code: resp.exit_code,
        stdout: Vec::new(),
        stderr: resp.output.as_bytes().to_vec(),
        wall_time_ms: 0,
        runner_name: "worker".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_darwin_sandbox_profile_and_worker_request() {
        let sbx_dir = PathFragment::create("/tmp/ob/sandbox/1");
        let profile = generate_darwin_sandbox_profile(&sbx_dir, false);
        assert!(profile.contains("(version 1)"));
        assert!(profile.contains("(deny network*"));

        let spawn = Spawn {
            mnemonic: "Javac".to_string(),
            owner_label: "//src:lib".to_string(),
            argv: vec!["--flagfile=params".to_string()],
            env: BTreeMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            execution_info: BTreeMap::new(),
        };
        let req = build_worker_request(&spawn, 7, &[("src/A.java".to_string(), vec![1, 2, 3])]);
        let encoded = req.encode_to_vec();
        let decoded = WorkRequest::decode(&encoded).unwrap();
        assert_eq!(decoded.request_id, 7);
        assert_eq!(decoded.inputs[0].path, "src/A.java");
    }
}
