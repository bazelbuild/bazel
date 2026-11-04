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

//! Bazel CLI Client (`bazel-rs`).
//! Mirrors `src/main/cpp/blaze.cc` (workspace discovery, `.bazelrc` loading, server lifecycle,
//! gRPC `CommandServer` connection, streaming `RunResponse`, and `ExecRequest` execution).

use bazel_options::{split_command_line, BazelrcConfig, StartupOptions};
use bazel_proto::command_server::{PingRequest, PingResponse, RunRequest, RunResponse, ServerInfo};
use bazel_proto::grpc_framing::{read_grpc_frame, write_grpc_frame};
use bazel_server::{
    create_default_runtime, serve_command_server, BlazeDirectories, BlazeRuntime, RpcMethod,
};
use bazel_util::ExitCode;
use bazel_vfs::{FileSystem, PathFragment};
use std::collections::HashSet;
use std::io::{self, BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const WORKSPACE_BOUNDARY_FILES: &[&str] = &[
    "MODULE.bazel",
    "REPO.bazel",
    "WORKSPACE.bazel",
    "WORKSPACE",
];

/// Walks up from `cwd` looking for a Bazel workspace boundary marker file.
pub fn find_workspace_root(fs: &dyn FileSystem, cwd: &PathFragment) -> PathFragment {
    let mut curr = Some(cwd.clone());
    while let Some(dir) = curr {
        for marker in WORKSPACE_BOUNDARY_FILES {
            if fs.exists(&dir.get_relative_str(marker), true) {
                return dir;
            }
        }
        curr = dir.parent_directory();
    }
    cwd.clone()
}

/// Sends a `PingRequest` to verify if an existing server is alive and matches `request_cookie`.
pub fn ping_server(address: &str, request_cookie: &str, expected_response_cookie: &str) -> bool {
    let Ok(stream) = TcpStream::connect(address) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return false,
    });
    let mut writer = BufWriter::new(stream);

    let req = PingRequest {
        cookie: request_cookie.to_string(),
    };
    let mut payload = vec![RpcMethod::Ping as u8];
    payload.extend_from_slice(&req.encode_to_vec());
    if write_grpc_frame(&mut writer, &payload).is_err() {
        return false;
    }
    let Ok(Some(resp_bytes)) = read_grpc_frame(&mut reader) else {
        return false;
    };
    let Ok(resp) = PingResponse::decode(&resp_bytes) else {
        return false;
    };
    resp.cookie == expected_response_cookie
}

/// Sends a `RunRequest` over the `CommandServer` stream and forwards `standard_output` and
/// `standard_error` chunks to the provided writers.
pub fn send_run_request<O: Write, E: Write>(
    address: &str,
    req: &RunRequest,
    expected_response_cookie: &str,
    stdout_writer: &mut O,
    stderr_writer: &mut E,
) -> io::Result<RunResponse> {
    let stream = TcpStream::connect(address)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = BufWriter::new(stream);

    let mut frame = vec![RpcMethod::Run as u8];
    frame.extend_from_slice(&req.encode_to_vec());
    write_grpc_frame(&mut writer, &frame)?;

    let mut final_response = RunResponse {
        exit_code: ExitCode::BLAZE_INTERNAL_ERROR.numeric_exit_code(),
        finished: true,
        ..Default::default()
    };

    while let Some(resp_bytes) = read_grpc_frame(&mut reader)? {
        let resp = RunResponse::decode(&resp_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        if !expected_response_cookie.is_empty() && resp.cookie != expected_response_cookie {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Server response cookie mismatch",
            ));
        }
        if !resp.standard_output.is_empty() {
            stdout_writer.write_all(&resp.standard_output)?;
            stdout_writer.flush()?;
        }
        if !resp.standard_error.is_empty() {
            stderr_writer.write_all(&resp.standard_error)?;
            stderr_writer.flush()?;
        }
        if resp.finished {
            final_response = resp;
            break;
        }
    }

    Ok(final_response)
}

/// Reads `$OUTPUT_BASE/server/server_info.rawproto` if present.
pub fn read_existing_server_info(
    fs: &dyn FileSystem,
    output_base: &PathFragment,
) -> Option<ServerInfo> {
    let info_path = output_base
        .get_relative_str("server")
        .get_relative_str("server_info.rawproto");
    let bytes = fs.read_file(&info_path).ok()?;
    ServerInfo::decode(&bytes).ok()
}

/// Entrypoint for running a CLI invocation and returning the numeric exit code.
pub fn execute_cli<O: Write, E: Write>(
    raw_args: &[String],
    cwd: &PathFragment,
    fs: Arc<dyn FileSystem>,
    stdout_writer: &mut O,
    stderr_writer: &mut E,
) -> i32 {
    // Internal flag used when spawning the background server process
    if raw_args.first().map(|s| s.as_str()) == Some("--internal_server_daemon") {
        return run_background_daemon(&raw_args[1..]);
    }

    let (mut startup, command, raw_cmd_args) = match split_command_line(raw_args) {
        Ok(res) => res,
        Err(e) => {
            let _ = writeln!(stderr_writer, "ERROR: {e}");
            return ExitCode::COMMAND_LINE_ERROR.numeric_exit_code();
        }
    };

    // Fast path for `bazel --version` (does not start server or load .bazelrc, matching C++ client & Bazelisk expectations)
    if command == "version" && raw_cmd_args.iter().any(|a| a == "--startup_version") {
        let _ = writeln!(stdout_writer, "bazel {}", bazel_server::build_label());
        return ExitCode::SUCCESS.numeric_exit_code();
    }

    let workspace_dir = find_workspace_root(fs.as_ref(), cwd);

    // Load .bazelrc files unless --ignore_all_rc_files is set
    let mut rc_config = BazelrcConfig::default();
    let mut visited_rcs = HashSet::new();
    if !startup.ignore_all_rc_files {
        if startup.workspace_rc {
            let ws_rc = workspace_dir.get_relative_str(".bazelrc");
            if let Ok(bytes) = fs.read_file(&ws_rc) {
                visited_rcs.insert(ws_rc.clone());
                let content = String::from_utf8_lossy(&bytes);
                let _ = rc_config.parse_str(
                    &content,
                    ws_rc.path_string(),
                    &workspace_dir,
                    fs.as_ref(),
                    &mut visited_rcs,
                );
            }
        }
        for custom_rc in startup.bazelrc_files.clone() {
            if custom_rc == "/dev/null" {
                continue;
            }
            let rc_pf = PathFragment::create(&custom_rc);
            let full_rc = if rc_pf.is_absolute() {
                rc_pf
            } else {
                workspace_dir.get_relative(&rc_pf)
            };
            if let Ok(bytes) = fs.read_file(&full_rc) {
                visited_rcs.insert(full_rc.clone());
                let content = String::from_utf8_lossy(&bytes);
                if let Err(e) = rc_config.parse_str(
                    &content,
                    full_rc.path_string(),
                    &workspace_dir,
                    fs.as_ref(),
                    &mut visited_rcs,
                ) {
                    let _ = writeln!(stderr_writer, "ERROR: {e}");
                    return ExitCode::COMMAND_LINE_ERROR.numeric_exit_code();
                }
            }
        }
        if let Some(startup_rc_opts) = rc_config.sections.get("startup") {
            for (flag, src) in startup_rc_opts {
                let _ = startup.apply_flag(flag, src);
            }
        }
    }

    let effective_cmd = if command.is_empty() {
        "help".to_string()
    } else {
        command
    };

    let rc_command = if effective_cmd == "canonicalize-flags" {
        raw_cmd_args
            .iter()
            .find_map(|a| a.strip_prefix("--for_command="))
            .unwrap_or("build")
    } else {
        effective_cmd.as_str()
    };

    let expanded_cmd_args = match rc_config.expand_command_args(rc_command, &raw_cmd_args) {
        Ok(args) => args,
        Err(e) => {
            let _ = writeln!(stderr_writer, "ERROR: {e}");
            return ExitCode::COMMAND_LINE_ERROR.numeric_exit_code();
        }
    };

    let user = std::env::var("USER").unwrap_or_else(|_| "bazel".to_string());
    let home = std::env::var("HOME").ok();
    let output_user_root = startup.resolve_output_user_root(&user, home.as_deref());
    let output_base = startup.resolve_output_base(&workspace_dir, &user, home.as_deref());
    let install_base =
        startup.resolve_install_base("bazel-rs-10.0.0-install-key", &user, home.as_deref());

    let mut full_argv_bytes = Vec::with_capacity(1 + expanded_cmd_args.len());
    full_argv_bytes.push(effective_cmd.as_bytes().to_vec());
    for a in &expanded_cmd_args {
        full_argv_bytes.push(a.as_bytes().to_vec());
    }

    let mut run_req = RunRequest {
        cookie: "batch-cookie".to_string(),
        arg: full_argv_bytes,
        block_for_lock: startup.block_for_lock,
        client_description: format!("pid={}", std::process::id()),
        invocation_policy: String::new(),
        startup_options: startup.raw_tagged_options.clone(),
        preemptible: false,
        command_extensions: Vec::new(),
        quiet: startup.quiet,
        block_for_lock_timeout_ms: None,
    };

    // If not in batch mode and using LocalFileSystem, check if a server daemon is already running
    if !startup.batch {
        if let Some(info) = read_existing_server_info(fs.as_ref(), &output_base) {
            if ping_server(&info.address, &info.request_cookie, &info.response_cookie) {
                run_req.cookie = info.request_cookie.clone();
                if let Ok(resp) = send_run_request(
                    &info.address,
                    &run_req,
                    &info.response_cookie,
                    stdout_writer,
                    stderr_writer,
                ) {
                    return resp.exit_code;
                }
            }
        }

        // Try starting an in-process or background server over localhost TCP
        if let Ok(listener) = TcpListener::bind(("127.0.0.1", 0)) {
            let req_cookie = "cookie-req-bazel-rs".to_string();
            let resp_cookie = "cookie-resp-bazel-rs".to_string();
            run_req.cookie = req_cookie.clone();

            // If we can spawn a detached daemon binary from current_exe, do so; otherwise serve in a thread
            let addr = listener.local_addr().unwrap().to_string();
            let dirs = BlazeDirectories {
                workspace_dir: workspace_dir.clone(),
                output_base: output_base.clone(),
                install_base: install_base.clone(),
                output_user_root: output_user_root.clone(),
            };
            let runtime = Arc::new(BlazeRuntime::new(
                dirs,
                Arc::clone(&fs),
                std::process::id(),
            ));
            let rt_clone = Arc::clone(&runtime);
            let req_c = req_cookie.clone();
            let resp_c = resp_cookie.clone();

            let server_handle = thread::spawn(move || {
                // Handle single request in embedded server mode if background daemon isn't detached
                let _ = serve_command_server(listener, rt_clone, req_c, resp_c);
            });

            // Wait briefly for server status files and connect
            for _ in 0..20 {
                if ping_server(&addr, &req_cookie, &resp_cookie) {
                    break;
                }
                thread::sleep(Duration::from_millis(5));
            }

            if let Ok(resp) = send_run_request(
                &addr,
                &run_req,
                &resp_cookie,
                stdout_writer,
                stderr_writer,
            ) {
                // Signal shutdown for the embedded single-shot server thread
                runtime
                    .shutdown_requested
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                let _ = TcpStream::connect(&addr);
                let _ = server_handle.join();
                return resp.exit_code;
            }
        }
    }

    // Fallback / `--batch` mode execution
    let dirs = BlazeDirectories {
        workspace_dir,
        output_base,
        install_base,
        output_user_root,
    };
    let runtime = BlazeRuntime::new(dirs, fs, std::process::id());
    let resp = runtime.dispatch_command(&run_req, "batch-cookie");
    let _ = stdout_writer.write_all(&resp.standard_output);
    let _ = stderr_writer.write_all(&resp.standard_error);
    resp.exit_code
}

fn run_background_daemon(args: &[String]) -> i32 {
    let mut workspace_dir = PathFragment::create("/");
    let mut output_base = PathFragment::create("/tmp/bazel-rs-ob");
    let mut install_base = PathFragment::create("/tmp/bazel-rs-ib");
    let mut output_user_root = PathFragment::create("/tmp/bazel-rs-root");
    let mut request_cookie = "cookie-req".to_string();
    let mut response_cookie = "cookie-resp".to_string();

    for arg in args {
        if let Some(v) = arg.strip_prefix("--workspace_directory=") {
            workspace_dir = PathFragment::create(v);
        } else if let Some(v) = arg.strip_prefix("--output_base=") {
            output_base = PathFragment::create(v);
        } else if let Some(v) = arg.strip_prefix("--install_base=") {
            install_base = PathFragment::create(v);
        } else if let Some(v) = arg.strip_prefix("--output_user_root=") {
            output_user_root = PathFragment::create(v);
        } else if let Some(v) = arg.strip_prefix("--request_cookie=") {
            request_cookie = v.to_string();
        } else if let Some(v) = arg.strip_prefix("--response_cookie=") {
            response_cookie = v.to_string();
        }
    }

    let Ok(listener) = TcpListener::bind(("127.0.0.1", 0)) else {
        return ExitCode::LOCAL_ENVIRONMENTAL_ERROR.numeric_exit_code();
    };
    let runtime = create_default_runtime(workspace_dir, output_base, install_base, output_user_root);
    match serve_command_server(listener, runtime, request_cookie, response_cookie) {
        Ok(()) => 0,
        Err(_) => ExitCode::LOCAL_ENVIRONMENTAL_ERROR.numeric_exit_code(),
    }
}

pub fn spawn_detached_daemon(
    exe_path: &std::path::Path,
    startup: &StartupOptions,
    workspace_dir: &PathFragment,
    output_base: &PathFragment,
    install_base: &PathFragment,
    output_user_root: &PathFragment,
    request_cookie: &str,
    response_cookie: &str,
) -> io::Result<u32> {
    let _ = startup;
    let child = Command::new(exe_path)
        .arg("--internal_server_daemon")
        .arg(format!("--workspace_directory={workspace_dir}"))
        .arg(format!("--output_base={output_base}"))
        .arg(format!("--install_base={install_base}"))
        .arg(format!("--output_user_root={output_user_root}"))
        .arg(format!("--request_cookie={request_cookie}"))
        .arg(format!("--response_cookie={response_cookie}"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    Ok(child.id())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bazel_vfs::InMemoryFileSystem;

    #[test]
    fn test_end_to_end_cli_commands() {
        let fs = Arc::new(InMemoryFileSystem::default());
        let ws = PathFragment::create("/workspace");
        fs.create_directory_and_parents(&ws).unwrap();
        fs.write_file(
            &ws.get_relative_str("MODULE.bazel"),
            b"module(name = \"my_ws\", version = \"1.0.0\")\n",
        )
        .unwrap();
        fs.write_file(
            &ws.get_relative_str(".bazelrc"),
            b"build:opt -c opt\n",
        )
        .unwrap();

        let pkg = ws.get_relative_str("app");
        fs.create_directory_and_parents(&pkg).unwrap();
        fs.write_file(
            &pkg.get_relative_str("BUILD.bazel"),
            b"cc_binary(name = \"hello\", srcs = [\"hello.cc\"])\n",
        )
        .unwrap();

        // 1. Test `bazel version`
        let mut out = Vec::new();
        let mut err = Vec::new();
        let code = execute_cli(
            &["--batch".to_string(), "version".to_string()],
            &ws,
            Arc::clone(&fs) as Arc<dyn FileSystem>,
            &mut out,
            &mut err,
        );
        assert_eq!(code, 0);
        assert!(String::from_utf8_lossy(&out).contains("Build label: 10.0.0-prerelease"));

        // 2. Test `bazel canonicalize-flags --config=opt`
        out.clear();
        err.clear();
        let code = execute_cli(
            &[
                "--batch".to_string(),
                "canonicalize-flags".to_string(),
                "--config=opt".to_string(),
            ],
            &ws,
            Arc::clone(&fs) as Arc<dyn FileSystem>,
            &mut out,
            &mut err,
        );
        assert_eq!(code, 0);
        assert_eq!(
            String::from_utf8_lossy(&out).trim(),
            "--compilation_mode=opt"
        );

        // 3. Test `bazel query //app/...`
        out.clear();
        err.clear();
        let code = execute_cli(
            &[
                "--batch".to_string(),
                "query".to_string(),
                "//app/...".to_string(),
            ],
            &ws,
            Arc::clone(&fs) as Arc<dyn FileSystem>,
            &mut out,
            &mut err,
        );
        assert_eq!(code, 0);
        assert_eq!(String::from_utf8_lossy(&out).trim(), "//app:hello");
    }
}
