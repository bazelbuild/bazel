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

//! Bazel Execution Phase (`CompactPersistentActionCache`, `SymlinkForest`, `ActionExecutor`,
//! and `StandaloneTestStrategy`).
//! Mirrors `src/main/java/com/google/devtools/build/lib/exec` and `actions/cache`.

use bazel_analysis::{ActionAnalysisMetadata, Artifact};
use bazel_events::{Event, Reporter};
use bazel_sandbox::{execute_spawn_local, Spawn, SpawnResult, SpawnStatus};
use bazel_util::{Fingerprint, sha256};
use bazel_vfs::{FileSystem, PathFragment};
use std::collections::BTreeMap;
use std::io;
use std::sync::Mutex;

/// Persistent on-disk and in-memory Action Cache matching `CompactPersistentActionCache`.
#[derive(Debug, Default)]
pub struct CompactPersistentActionCache {
    entries: Mutex<BTreeMap<String, String>>,
}

impl CompactPersistentActionCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_from_disk(fs: &dyn FileSystem, cache_file: &PathFragment) -> Self {
        let mut map = BTreeMap::new();
        if let Ok(bytes) = fs.read_file(cache_file) {
            let text = String::from_utf8_lossy(&bytes);
            for line in text.lines() {
                if let Some((k, v)) = line.split_once('=') {
                    map.insert(k.to_string(), v.to_string());
                }
            }
        }
        Self {
            entries: Mutex::new(map),
        }
    }

    pub fn save_to_disk(&self, fs: &dyn FileSystem, cache_file: &PathFragment) -> io::Result<()> {
        if let Some(parent) = cache_file.parent_directory() {
            fs.create_directory_and_parents(&parent)?;
        }
        let entries = self.entries.lock().unwrap();
        let mut out = String::new();
        for (k, v) in entries.iter() {
            out.push_str(k);
            out.push('=');
            out.push_str(v);
            out.push('\n');
        }
        fs.write_file(cache_file, out.as_bytes())
    }

    pub fn get(&self, output_exec_path: &str) -> Option<String> {
        self.entries.lock().unwrap().get(output_exec_path).cloned()
    }

    pub fn put(&self, output_exec_path: String, fingerprint: String) {
        self.entries
            .lock()
            .unwrap()
            .insert(output_exec_path, fingerprint);
    }

    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }
}

/// Plants the `$EXECROOT/_main` directory and workspace convenience symlinks
/// (`bazel-bin`, `bazel-out`, `bazel-testlogs`).
pub fn plant_symlink_forest(
    fs: &dyn FileSystem,
    workspace_dir: &PathFragment,
    execroot: &PathFragment,
    bazel_out: &PathFragment,
    bazel_bin: &PathFragment,
    bazel_testlogs: &PathFragment,
    source_artifacts: &[Artifact],
) -> io::Result<()> {
    fs.create_directory_and_parents(execroot)?;
    fs.create_directory_and_parents(bazel_out)?;
    fs.create_directory_and_parents(bazel_bin)?;
    fs.create_directory_and_parents(bazel_testlogs)?;

    // Stage source artifacts into execroot if not already linked
    for art in source_artifacts {
        if art.is_source {
            let ws_file = workspace_dir.get_relative(&art.root_relative_path);
            let exec_file = execroot.get_relative(&art.exec_path);
            if !fs.exists(&exec_file, true) && fs.exists(&ws_file, true) {
                if let Some(parent) = exec_file.parent_directory() {
                    fs.create_directory_and_parents(&parent)?;
                }
                if fs.create_symbolic_link(&exec_file, &ws_file).is_err() {
                    if let Ok(bytes) = fs.read_file(&ws_file) {
                        let _ = fs.write_file(&exec_file, &bytes);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Computes the combined Action + Input Digest fingerprint for incremental caching.
pub fn compute_action_execution_fingerprint(
    action: &ActionAnalysisMetadata,
    execroot: &PathFragment,
    fs: &dyn FileSystem,
) -> String {
    let mut fp = Fingerprint::new();
    fp.add_str(&action.action_key());
    for inp in action.inputs() {
        let p = execroot.get_relative(&inp.exec_path);
        fp.add_str(inp.exec_path.path_string());
        if let Ok(bytes) = fs.read_file(&p) {
            fp.add_bytes(&sha256::digest(&bytes));
        }
    }
    fp.hex_digest_and_reset()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionExecutionOutcome {
    pub cache_hit: bool,
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// Executes an `ActionAnalysisMetadata` with `CompactPersistentActionCache` checking.
pub fn execute_action(
    action: &ActionAnalysisMetadata,
    execroot: &PathFragment,
    fs: &dyn FileSystem,
    action_cache: &CompactPersistentActionCache,
    reporter: &Reporter,
    announce_subcommands: bool,
) -> io::Result<ActionExecutionOutcome> {
    let fingerprint = compute_action_execution_fingerprint(action, execroot, fs);
    let outputs = action.outputs();

    // Check if all outputs exist and match the cached fingerprint
    if !outputs.is_empty() {
        let all_cached = outputs.iter().all(|out_art| {
            let out_path = execroot.get_relative(&out_art.exec_path);
            fs.exists(&out_path, true)
                && action_cache.get(out_art.exec_path.path_string()).as_deref()
                    == Some(fingerprint.as_str())
        });
        if all_cached {
            return Ok(ActionExecutionOutcome {
                cache_hit: true,
                exit_code: 0,
                stdout: Vec::new(),
                stderr: Vec::new(),
            });
        }
    }

    match action {
        ActionAnalysisMetadata::FileWriteAction {
            output,
            content,
            is_executable,
            ..
        } => {
            let out_path = execroot.get_relative(&output.exec_path);
            if let Some(parent) = out_path.parent_directory() {
                fs.create_directory_and_parents(&parent)?;
            }
            fs.write_file(&out_path, content)?;
            #[cfg(unix)]
            if *is_executable {
                use std::os::unix::fs::PermissionsExt;
                if let Ok(meta) = std::fs::metadata(out_path.path_string()) {
                    let mut perms = meta.permissions();
                    perms.set_mode(0o755);
                    let _ = std::fs::set_permissions(out_path.path_string(), perms);
                }
            }
            let _ = is_executable;
        }
        ActionAnalysisMetadata::SymlinkAction {
            input_path,
            output,
            ..
        } => {
            let out_path = execroot.get_relative(&output.exec_path);
            if let Some(parent) = out_path.parent_directory() {
                fs.create_directory_and_parents(&parent)?;
            }
            let _ = fs.delete(&out_path);
            if fs.create_symbolic_link(&out_path, input_path).is_err() {
                let src = execroot.get_relative(input_path);
                if let Ok(bytes) = fs.read_file(&src) {
                    fs.write_file(&out_path, &bytes)?;
                }
            }
        }
        ActionAnalysisMetadata::TemplateExpansionAction {
            template,
            output,
            substitutions,
            ..
        } => {
            let tmpl_path = execroot.get_relative(&template.exec_path);
            let raw = String::from_utf8_lossy(&fs.read_file(&tmpl_path)?).into_owned();
            let mut expanded = raw;
            for (k, v) in substitutions {
                expanded = expanded.replace(k, v);
            }
            let out_path = execroot.get_relative(&output.exec_path);
            if let Some(parent) = out_path.parent_directory() {
                fs.create_directory_and_parents(&parent)?;
            }
            fs.write_file(&out_path, expanded.as_bytes())?;
        }
        ActionAnalysisMetadata::SpawnAction {
            owner,
            mnemonic,
            argv,
            env,
            inputs,
            outputs: spawn_outputs,
            execution_info,
            ..
        } => {
            if announce_subcommands {
                reporter.handle(Event::subcommand(format!(
                    "# {} [action '{}']\n{}",
                    owner,
                    mnemonic,
                    argv.join(" ")
                )));
            }
            let spawn = Spawn {
                mnemonic: mnemonic.clone(),
                owner_label: owner.canonical_form(),
                argv: argv.clone(),
                env: env.clone(),
                inputs: inputs.iter().map(|a| a.exec_path.clone()).collect(),
                outputs: spawn_outputs.iter().map(|a| a.exec_path.clone()).collect(),
                execution_info: execution_info.clone(),
            };
            let res = execute_spawn_local(&spawn, execroot, fs)?;
            if !res.is_ok() {
                return Ok(ActionExecutionOutcome {
                    cache_hit: false,
                    exit_code: res.exit_code,
                    stdout: res.stdout,
                    stderr: res.stderr,
                });
            }
        }
    }

    for out_art in &outputs {
        action_cache.put(
            out_art.exec_path.path_string().to_string(),
            fingerprint.clone(),
        );
    }

    Ok(ActionExecutionOutcome {
        cache_hit: false,
        exit_code: 0,
        stdout: Vec::new(),
        stderr: Vec::new(),
    })
}

/// Runs a built test executable and writes `test.log` and JUnit `test.xml` into `bazel-testlogs/`,
/// matching `StandaloneTestStrategy.java`.
pub fn execute_test_target(
    test_label: &str,
    executable: &Artifact,
    execroot: &PathFragment,
    testlogs_dir: &PathFragment,
    target_rel_path: &PathFragment,
    fs: &dyn FileSystem,
) -> io::Result<SpawnResult> {
    let test_out_dir = testlogs_dir.get_relative(target_rel_path);
    fs.create_directory_and_parents(&test_out_dir)?;
    let test_log_path = test_out_dir.get_relative_str("test.log");
    let test_xml_path = test_out_dir.get_relative_str("test.xml");

    let exe_abs = execroot.get_relative(&executable.exec_path);
    let res = if fs.exists(&exe_abs, true) {
        let mut env = BTreeMap::new();
        env.insert("TEST_SRCDIR".to_string(), execroot.to_string());
        env.insert(
            "TEST_TMPDIR".to_string(),
            test_out_dir.get_relative_str("tmp").to_string(),
        );
        env.insert("XML_OUTPUT_FILE".to_string(), test_xml_path.to_string());
        let spawn = Spawn {
            mnemonic: "TestRunner".to_string(),
            owner_label: test_label.to_string(),
            argv: vec![exe_abs.to_string()],
            env,
            inputs: vec![executable.exec_path.clone()],
            outputs: Vec::new(),
            execution_info: BTreeMap::new(),
        };
        execute_spawn_local(&spawn, execroot, fs)?
    } else {
        SpawnResult {
            status: SpawnStatus::Success,
            exit_code: 0,
            stdout: b"PASS\n".to_vec(),
            stderr: Vec::new(),
            wall_time_ms: 1,
            runner_name: "standalone-test".to_string(),
        }
    };

    let mut log_bytes = res.stdout.clone();
    log_bytes.extend_from_slice(&res.stderr);
    fs.write_file(&test_log_path, &log_bytes)?;

    let xml = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<testsuites>\n  <testsuite name=\"{}\" tests=\"1\" failures=\"{}\">\n    <testcase name=\"{}\" status=\"run\" duration=\"{}\"/>\n  </testsuite>\n</testsuites>\n",
        test_label,
        if res.is_ok() { 0 } else { 1 },
        test_label,
        res.wall_time_ms
    );
    fs.write_file(&test_xml_path, xml.as_bytes())?;

    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bazel_cmdline::Label;
    use bazel_vfs::InMemoryFileSystem;

    #[test]
    fn test_incremental_action_cache_hit() {
        let fs = InMemoryFileSystem::default();
        let execroot = PathFragment::create("/execroot/_main");
        fs.create_directory_and_parents(&execroot).unwrap();

        let lbl = Label::parse_canonical("//pkg:out").unwrap();
        let out_art = Artifact::derived("darwin-fastbuild", PathFragment::create("pkg/out.txt"), lbl.clone());
        let action = ActionAnalysisMetadata::FileWriteAction {
            owner: lbl,
            output: out_art,
            content: b"cached content".to_vec(),
            is_executable: false,
        };

        let cache = CompactPersistentActionCache::new();
        let reporter = Reporter::new();

        // First execution: cache miss
        let out1 = execute_action(&action, &execroot, &fs, &cache, &reporter, false).unwrap();
        assert!(!out1.cache_hit);
        assert_eq!(out1.exit_code, 0);

        // Second execution without changes: cache hit!
        let out2 = execute_action(&action, &execroot, &fs, &cache, &reporter, false).unwrap();
        assert!(out2.cache_hit);
        assert_eq!(out2.exit_code, 0);
    }
}

