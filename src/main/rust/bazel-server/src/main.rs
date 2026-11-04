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

use bazel_server::{create_default_runtime, serve_command_server};
use bazel_vfs::PathFragment;
use std::net::TcpListener;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut workspace_dir = PathFragment::create(
        &std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "/".to_string()),
    );
    let mut output_base = PathFragment::create("/tmp/bazel-rs-ob");
    let mut install_base = PathFragment::create("/tmp/bazel-rs-ib");
    let mut output_user_root = PathFragment::create("/tmp/bazel-rs-root");
    let mut port: u16 = 0;
    let mut request_cookie = "default_req_cookie".to_string();
    let mut response_cookie = "default_resp_cookie".to_string();

    for arg in &args {
        if let Some(v) = arg.strip_prefix("--workspace_directory=") {
            workspace_dir = PathFragment::create(v);
        } else if let Some(v) = arg.strip_prefix("--output_base=") {
            output_base = PathFragment::create(v);
        } else if let Some(v) = arg.strip_prefix("--install_base=") {
            install_base = PathFragment::create(v);
        } else if let Some(v) = arg.strip_prefix("--output_user_root=") {
            output_user_root = PathFragment::create(v);
        } else if let Some(v) = arg.strip_prefix("--port=") {
            port = v.parse().unwrap_or(0);
        } else if let Some(v) = arg.strip_prefix("--request_cookie=") {
            request_cookie = v.to_string();
        } else if let Some(v) = arg.strip_prefix("--response_cookie=") {
            response_cookie = v.to_string();
        }
    }

    let listener = TcpListener::bind(("127.0.0.1", port)).expect("Failed to bind CommandServer");
    let runtime = create_default_runtime(workspace_dir, output_base, install_base, output_user_root);
    if let Err(e) = serve_command_server(listener, runtime, request_cookie, response_cookie) {
        eprintln!("FATAL: CommandServer failed: {e}");
        std::process::exit(36);
    }
}

