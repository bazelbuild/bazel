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

//! Remote Execution API v2 (REAPI) client, Disk Cache (`--disk_cache`), and Chrome Trace Profiler (`--profile`).
//! Mirrors `src/main/java/com/google/devtools/build/lib/remote` and `profiler`.

use bazel_proto::remote_execution::{Action, ActionResult, Command, Digest, OutputFile};
use bazel_util::sha256;
use bazel_vfs::{FileSystem, PathFragment};
use std::io;
use std::sync::Mutex;

/// Computes the REAPI v2 `Digest` (`SHA-256` hex + size in bytes) of a blob.
pub fn compute_digest(blob: &[u8]) -> Digest {
    Digest {
        hash: sha256::hex_digest(blob),
        size_bytes: blob.len() as i64,
    }
}

/// Computes the REAPI v2 `Command` and `Action` digests for a spawn.
pub fn build_reapi_action(
    argv: &[String],
    output_files: &[String],
    input_root_digest: Digest,
) -> (Command, Digest, Action, Digest) {
    let cmd = Command {
        arguments: argv.to_vec(),
        environment_variables: Vec::new(),
        output_files: output_files.to_vec(),
        output_directories: Vec::new(),
        working_directory: String::new(),
    };
    let cmd_bytes = cmd.encode_to_vec();
    let cmd_digest = compute_digest(&cmd_bytes);

    let action = Action {
        command_digest: Some(cmd_digest.clone()),
        input_root_digest: Some(input_root_digest),
        do_not_cache: false,
    };
    let action_bytes = action.encode_to_vec();
    let action_digest = compute_digest(&action_bytes);

    (cmd, cmd_digest, action, action_digest)
}

/// Combined Disk Cache (`--disk_cache`) and Remote Action Cache (`ac/` + `cas/` layout).
pub struct DiskCacheClient {
    pub cache_root: PathFragment,
}

impl DiskCacheClient {
    pub fn new(cache_root: PathFragment) -> Self {
        Self { cache_root }
    }

    fn ac_path(&self, action_hash: &str) -> PathFragment {
        let prefix = if action_hash.len() >= 2 {
            &action_hash[..2]
        } else {
            "00"
        };
        self.cache_root
            .get_relative_str("ac")
            .get_relative_str(prefix)
            .get_relative_str(action_hash)
    }

    fn cas_path(&self, blob_hash: &str) -> PathFragment {
        let prefix = if blob_hash.len() >= 2 {
            &blob_hash[..2]
        } else {
            "00"
        };
        self.cache_root
            .get_relative_str("cas")
            .get_relative_str(prefix)
            .get_relative_str(blob_hash)
    }

    pub fn get_action_result(
        &self,
        fs: &dyn FileSystem,
        action_digest: &Digest,
    ) -> Option<ActionResult> {
        let bytes = fs.read_file(&self.ac_path(&action_digest.hash)).ok()?;
        ActionResult::decode(&bytes).ok()
    }

    pub fn put_action_result(
        &self,
        fs: &dyn FileSystem,
        action_digest: &Digest,
        result: &ActionResult,
    ) -> io::Result<()> {
        let path = self.ac_path(&action_digest.hash);
        if let Some(parent) = path.parent_directory() {
            fs.create_directory_and_parents(&parent)?;
        }
        fs.write_file(&path, &result.encode_to_vec())
    }

    pub fn upload_blob(&self, fs: &dyn FileSystem, data: &[u8]) -> io::Result<Digest> {
        let digest = compute_digest(data);
        let path = self.cas_path(&digest.hash);
        if let Some(parent) = path.parent_directory() {
            fs.create_directory_and_parents(&parent)?;
        }
        fs.write_file(&path, data)?;
        Ok(digest)
    }

    pub fn download_blob(&self, fs: &dyn FileSystem, digest: &Digest) -> io::Result<Vec<u8>> {
        fs.read_file(&self.cas_path(&digest.hash))
    }

    pub fn cache_outputs_from_execroot(
        &self,
        fs: &dyn FileSystem,
        execroot: &PathFragment,
        action_digest: &Digest,
        output_rel_paths: &[PathFragment],
        exit_code: i32,
    ) -> io::Result<ActionResult> {
        let mut output_files = Vec::new();
        for rel in output_rel_paths {
            let full = execroot.get_relative(rel);
            if let Ok(bytes) = fs.read_file(&full) {
                let blob_digest = self.upload_blob(fs, &bytes)?;
                output_files.push(OutputFile {
                    path: rel.path_string().to_string(),
                    digest: Some(blob_digest),
                    is_executable: false,
                });
            }
        }
        let ar = ActionResult {
            output_files,
            exit_code,
            stdout_raw: Vec::new(),
            stderr_raw: Vec::new(),
        };
        self.put_action_result(fs, action_digest, &ar)?;
        Ok(ar)
    }
}

#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub name: String,
    pub category: String,
    pub start_us: u64,
    pub duration_us: u64,
    pub tid: u32,
}

/// Chrome Trace Format JSON profiler (`--profile`), compatible with `bazel analyze-profile`.
#[derive(Debug, Default)]
pub struct Profiler {
    events: Mutex<Vec<TraceEvent>>,
}

impl Profiler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&self, name: impl Into<String>, category: impl Into<String>, start_us: u64, duration_us: u64) {
        self.events.lock().unwrap().push(TraceEvent {
            name: name.into(),
            category: category.into(),
            start_us,
            duration_us,
            tid: 1,
        });
    }

    pub fn write_profile(&self, fs: &dyn FileSystem, path: &PathFragment) -> io::Result<()> {
        let events = self.events.lock().unwrap();
        let mut json = String::from("{\"traceEvents\":[\n");
        for (i, ev) in events.iter().enumerate() {
            if i > 0 {
                json.push_str(",\n");
            }
            json.push_str(&format!(
                "{{\"name\":\"{}\",\"cat\":\"{}\",\"ph\":\"X\",\"ts\":{},\"dur\":{},\"pid\":1,\"tid\":{}}}",
                ev.name, ev.category, ev.start_us, ev.duration_us, ev.tid
            ));
        }
        json.push_str("\n]}\n");
        fs.write_file(path, json.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bazel_vfs::InMemoryFileSystem;

    #[test]
    fn test_disk_cache_cas_and_ac_roundtrip() {
        let fs = InMemoryFileSystem::default();
        let cache = DiskCacheClient::new(PathFragment::create("/cache"));
        let execroot = PathFragment::create("/execroot/_main");
        fs.create_directory_and_parents(&execroot.get_relative_str("out"))
            .unwrap();
        let out_rel = PathFragment::create("out/hello.txt");
        fs.write_file(&execroot.get_relative(&out_rel), b"hello world")
            .unwrap();

        let input_digest = compute_digest(b"input-tree");
        let (_cmd, _cd, _act, act_digest) = build_reapi_action(
            &["echo".to_string(), "hello".to_string()],
            &["out/hello.txt".to_string()],
            input_digest,
        );

        let stored = cache
            .cache_outputs_from_execroot(&fs, &execroot, &act_digest, &[out_rel], 0)
            .unwrap();
        assert_eq!(stored.output_files.len(), 1);

        let fetched = cache.get_action_result(&fs, &act_digest).unwrap();
        assert_eq!(fetched, stored);
        let blob = cache
            .download_blob(&fs, fetched.output_files[0].digest.as_ref().unwrap())
            .unwrap();
        assert_eq!(blob, b"hello world");
    }
}

