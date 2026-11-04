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

use bazel_client::execute_cli;
use bazel_vfs::{LocalFileSystem, PathFragment};
use std::io;
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cwd_str = std::env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "/".to_string());
    let cwd = PathFragment::create(&cwd_str);
    let fs = Arc::new(LocalFileSystem::default());

    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    let exit_code = execute_cli(&args, &cwd, fs, &mut stdout, &mut stderr);
    std::process::exit(exit_code);
}

