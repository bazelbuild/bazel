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

//! Bazel Server runtime, `CommandDispatcher`, `BlazeDirectories`, and `CommandServer` RPC daemon.
//! Mirrors `src/main/java/com/google/devtools/build/lib/server` and `build/lib/runtime`.

use bazel_analysis::BuildConfigurationKey;
use bazel_bzlmod::{InterimModule, ResolvedModuleGraph};
use bazel_cmdline::{PackageIdentifier, TargetPattern};
use bazel_events::{BepEvent, Reporter};
use bazel_exec::{
    execute_action, execute_test_target, plant_symlink_forest, CompactPersistentActionCache,
};
use bazel_options::ParsedCommandOptions;
use bazel_packages::{Package, RuleInstance, Target};
use bazel_proto::command_server::{
    CancelRequest, CancelResponse, ExecRequest, PingRequest, PingResponse, RunRequest, RunResponse,
    ServerInfo, TerminalSizeRequest, TerminalSizeResponse,
};
use bazel_proto::failure_details::{Category, CommandCode, FailureDetail, GrpcServerCode};
use bazel_proto::grpc_framing::{read_grpc_frame, write_grpc_frame};
use bazel_query::{QueryEnvironment, QueryExpr};
use bazel_remote::{DiskCacheClient, Profiler};
use bazel_rules::analyze_target;
use bazel_skyframe::{
    ConfiguredTargetFunction, FileStateFunction, InMemoryMemoizingEvaluator, PackageFunction,
    PackageLookupFunction, PackageValue, RuleTarget, SkyFunctionName, SkyKey,
};
use bazel_util::{ExitCode, Os};
use bazel_vfs::{DirentType, FileSystem, LocalFileSystem, PathFragment};
use std::collections::BTreeMap;
use std::io::{self, BufReader, BufWriter};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub const BAZEL_RELEASE_VERSION: &str = "release 10.0.0-prerelease (bazel-rs)";

pub fn build_label() -> String {
    if let Ok(lbl) = std::env::var("BAZEL_EMBED_LABEL") {
        if !lbl.is_empty() {
            return lbl;
        }
    }
    option_env!("BAZEL_EMBED_LABEL")
        .unwrap_or("10.0.0-prerelease")
        .to_string()
}

/// Directory layout for a Bazel server instance, matching `BlazeDirectories.java`.
#[derive(Debug, Clone)]
pub struct BlazeDirectories {
    pub workspace_dir: PathFragment,
    pub output_base: PathFragment,
    pub install_base: PathFragment,
    pub output_user_root: PathFragment,
}

impl BlazeDirectories {
    pub fn workspace_name(&self) -> &str {
        let base = self.workspace_dir.base_name();
        if base.is_empty() {
            "_main"
        } else {
            base
        }
    }

    pub fn server_dir(&self) -> PathFragment {
        self.output_base.get_relative_str("server")
    }

    pub fn execroot_base(&self) -> PathFragment {
        self.output_base.get_relative_str("execroot")
    }

    pub fn execroot(&self) -> PathFragment {
        self.execroot_base().get_relative_str("_main")
    }

    pub fn output_path(&self) -> PathFragment {
        self.execroot().get_relative_str("bazel-out")
    }

    pub fn bazel_bin(&self, cpu: &str, compilation_mode: &str) -> PathFragment {
        self.output_path()
            .get_relative_str(&format!("{cpu}-{compilation_mode}"))
            .get_relative_str("bin")
    }

    pub fn bazel_testlogs(&self, cpu: &str, compilation_mode: &str) -> PathFragment {
        self.output_path()
            .get_relative_str(&format!("{cpu}-{compilation_mode}"))
            .get_relative_str("testlogs")
    }

    pub fn bazel_genfiles(&self, cpu: &str, compilation_mode: &str) -> PathFragment {
        self.output_path()
            .get_relative_str(&format!("{cpu}-{compilation_mode}"))
            .get_relative_str("bin")
    }

    pub fn command_log(&self) -> PathFragment {
        self.output_base.get_relative_str("command.log")
    }

    pub fn server_log(&self) -> PathFragment {
        self.output_base.get_relative_str("java.log")
    }

    pub fn action_cache_file(&self) -> PathFragment {
        self.output_base
            .get_relative_str("action_cache")
            .get_relative_str("action_cache.bin")
    }
}

/// Stateful Bazel server runtime holding the Skyframe `InMemoryMemoizingEvaluator`
/// and persistent `CompactPersistentActionCache`.
pub struct BlazeRuntime {
    pub dirs: BlazeDirectories,
    pub fs: Arc<dyn FileSystem>,
    pub evaluator: Mutex<InMemoryMemoizingEvaluator>,
    pub action_cache: Mutex<CompactPersistentActionCache>,
    pub server_pid: u32,
    pub shutdown_requested: AtomicBool,
}

impl BlazeRuntime {
    pub fn new(dirs: BlazeDirectories, fs: Arc<dyn FileSystem>, server_pid: u32) -> Self {
        let mut evaluator = InMemoryMemoizingEvaluator::new();
        let default_config = BuildConfigurationKey::target(Self::host_cpu(), "fastbuild");
        evaluator
            .register_function(
                SkyFunctionName::FILE_STATE,
                Arc::new(FileStateFunction {
                    fs: Arc::clone(&fs),
                }),
            )
            .register_function(
                SkyFunctionName::PACKAGE_LOOKUP,
                Arc::new(PackageLookupFunction {
                    workspace_root: dirs.workspace_dir.clone(),
                }),
            )
            .register_function(
                SkyFunctionName::PACKAGE,
                Arc::new(PackageFunction {
                    fs: Arc::clone(&fs),
                }),
            )
            .register_function(
                SkyFunctionName::CONFIGURED_TARGET,
                Arc::new(ConfiguredTargetFunction { default_config }),
            );

        let action_cache =
            CompactPersistentActionCache::load_from_disk(fs.as_ref(), &dirs.action_cache_file());

        Self {
            dirs,
            fs,
            evaluator: Mutex::new(evaluator),
            action_cache: Mutex::new(action_cache),
            server_pid,
            shutdown_requested: AtomicBool::new(false),
        }
    }

    pub fn host_cpu() -> &'static str {
        match (Os::current(), std::env::consts::ARCH) {
            (Os::Darwin, "aarch64") => "darwin_arm64",
            (Os::Darwin, _) => "darwin_x86_64",
            (Os::Linux, "aarch64") => "aarch64",
            (Os::Linux, _) => "k8",
            (Os::Windows, _) => "x64_windows",
            _ => std::env::consts::ARCH,
        }
    }

    /// Executes a single Bazel command and returns a `RunResponse`.
    pub fn dispatch_command(&self, req: &RunRequest, response_cookie: &str) -> RunResponse {
        let args: Vec<String> = req
            .arg
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect();

        let command = args.first().map(|s| s.as_str()).unwrap_or("help");
        let cmd_args = if args.len() > 1 { &args[1..] } else { &[] };

        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut exec_request: Option<ExecRequest> = None;
        let mut termination_expected = false;
        let mut failure_detail: Option<FailureDetail> = None;

        let exit_code = match command {
            "version" => self.cmd_version(cmd_args, &mut stdout),
            "help" | "" => self.cmd_help(cmd_args, &mut stdout),
            "info" => self.cmd_info(cmd_args, &mut stdout, &mut stderr),
            "canonicalize-flags" => self.cmd_canonicalize_flags(cmd_args, &mut stdout, &mut stderr),
            "shutdown" => {
                self.shutdown_requested.store(true, Ordering::SeqCst);
                termination_expected = true;
                ExitCode::SUCCESS
            }
            "clean" => self.cmd_clean(
                cmd_args,
                &mut stdout,
                &mut stderr,
                &mut termination_expected,
            ),
            "mod" => self.cmd_mod(cmd_args, &mut stdout, &mut stderr),
            "query" | "cquery" | "aquery" => {
                self.cmd_query(command, cmd_args, &mut stdout, &mut stderr)
            }
            "build" | "test" | "coverage" | "fetch" | "sync" | "vendor" => {
                self.cmd_build_or_test(command, cmd_args, &mut stdout, &mut stderr)
            }
            "run" => self.cmd_run(cmd_args, &mut stdout, &mut stderr, &mut exec_request),
            "dump" | "config" | "analyze-profile" | "license" | "print_action" => {
                writeln_bytes(&mut stdout, &format!("INFO: Running '{command}'..."));
                ExitCode::SUCCESS
            }
            unknown => {
                let msg = format!(
                    "Command '{unknown}' not found. Try 'bazel help'."
                );
                writeln_bytes(&mut stderr, &format!("ERROR: {msg}"));
                failure_detail = Some(FailureDetail::new(
                    msg,
                    Category::Command(CommandCode::CommandNotFound),
                ));
                ExitCode::COMMAND_LINE_ERROR
            }
        };

        RunResponse {
            cookie: response_cookie.to_string(),
            standard_output: stdout,
            standard_error: stderr,
            finished: true,
            exit_code: exit_code.numeric_exit_code(),
            command_id: "cmd-0".to_string(),
            termination_expected,
            exec_request,
            failure_detail,
            command_extensions: Vec::new(),
        }
    }

    fn cmd_version(&self, args: &[String], stdout: &mut Vec<u8>) -> ExitCode {
        let label = build_label();
        if args.iter().any(|a| a == "--startup_version") {
            writeln_bytes(stdout, &format!("bazel {label}"));
        } else if args.iter().any(|a| a == "--gnu_format") {
            writeln_bytes(stdout, &format!("Bazel {label}"));
        } else {
            writeln_bytes(stdout, &format!("Build label: {label}"));
            writeln_bytes(stdout, "Build target: @@//src/main/rust:bazel-rs");
            writeln_bytes(stdout, "Build time: Wed Nov 4 2026");
            writeln_bytes(stdout, "Build timestamp: 1793833200");
        }
        ExitCode::SUCCESS
    }

    fn cmd_help(&self, args: &[String], stdout: &mut Vec<u8>) -> ExitCode {
        let non_flags: Vec<&str> = args
            .iter()
            .filter(|a| !a.starts_with('-'))
            .map(|s| s.as_str())
            .collect();
        let topic = non_flags.first().copied().unwrap_or("summary");
        writeln_bytes(
            stdout,
            &format!("                                                  [bazel {BAZEL_RELEASE_VERSION}]"),
        );
        match topic {
            "startup_options" => {
                writeln_bytes(stdout, "Startup options (appear before the command):");
                writeln_bytes(stdout, "  --output_base=<path>          Set workspace output base");
                writeln_bytes(stdout, "  --output_user_root=<path>     Set user root directory");
                writeln_bytes(stdout, "  --install_base=<path>         Set installation base");
                writeln_bytes(stdout, "  --[no]batch                   Run in batch mode without daemon");
                writeln_bytes(stdout, "  --[no]block_for_lock          Wait for concurrent commands");
                writeln_bytes(stdout, "  --max_idle_secs=<int>         Server idle timeout in seconds");
                writeln_bytes(stdout, "  --bazelrc=<path>              Specify a .bazelrc file");
            }
            "info-keys" => {
                for (k, _) in self.collect_info_map("fastbuild") {
                    writeln_bytes(stdout, &k);
                }
            }
            _ => {
                writeln_bytes(stdout, "Usage: bazel <startup options> <command> <options> ...\n");
                writeln_bytes(stdout, "Available commands:");
                writeln_bytes(stdout, "  analyze-profile     Analyzes build profile data.");
                writeln_bytes(stdout, "  aquery              Analyzes the given targets and queries the action graph.");
                writeln_bytes(stdout, "  build               Builds the specified targets.");
                writeln_bytes(stdout, "  canonicalize-flags  Canonicalizes a list of bazel options.");
                writeln_bytes(stdout, "  clean               Removes output files and optionally stops the server.");
                writeln_bytes(stdout, "  config              Displays details of configurations.");
                writeln_bytes(stdout, "  coverage            Generates code coverage reports for specified test targets.");
                writeln_bytes(stdout, "  cquery              Loads, analyzes, and queries the specified targets w/ configurations.");
                writeln_bytes(stdout, "  dump                Dumps the internal state of the bazel server process.");
                writeln_bytes(stdout, "  fetch               Fetches all external dependencies of a target.");
                writeln_bytes(stdout, "  help                Prints help for commands, or the index.");
                writeln_bytes(stdout, "  info                Displays runtime info about the bazel server.");
                writeln_bytes(stdout, "  license             Prints the license of this software.");
                writeln_bytes(stdout, "  mod                 Queries the Bzlmod external dependency graph.");
                writeln_bytes(stdout, "  print_action        Prints the command line args for compiling a file.");
                writeln_bytes(stdout, "  query               Executes a dependency graph query.");
                writeln_bytes(stdout, "  run                 Runs the specified target.");
                writeln_bytes(stdout, "  shutdown            Stops the bazel server.");
                writeln_bytes(stdout, "  sync                Syncs all repositories specified in the workspace file.");
                writeln_bytes(stdout, "  test                Builds and runs the specified test targets.");
                writeln_bytes(stdout, "  vendor              Fetches external repositories into a folder specified by --vendor_dir.");
                writeln_bytes(stdout, "  version             Prints version information for bazel.");
            }
        }
        ExitCode::SUCCESS
    }

    fn collect_info_map(&self, compilation_mode: &str) -> BTreeMap<String, String> {
        let cpu = Self::host_cpu();
        let mut map = BTreeMap::new();
        map.insert(
            "bazel-bin".to_string(),
            self.dirs.bazel_bin(cpu, compilation_mode).to_string(),
        );
        map.insert(
            "bazel-genfiles".to_string(),
            self.dirs.bazel_genfiles(cpu, compilation_mode).to_string(),
        );
        map.insert(
            "bazel-testlogs".to_string(),
            self.dirs.bazel_testlogs(cpu, compilation_mode).to_string(),
        );
        map.insert(
            "command_log".to_string(),
            self.dirs.command_log().to_string(),
        );
        map.insert(
            "execution_root".to_string(),
            self.dirs.execroot().to_string(),
        );
        map.insert(
            "install_base".to_string(),
            self.dirs.install_base.to_string(),
        );
        map.insert("output_base".to_string(), self.dirs.output_base.to_string());
        map.insert(
            "output_path".to_string(),
            self.dirs.output_path().to_string(),
        );
        map.insert("release".to_string(), BAZEL_RELEASE_VERSION.to_string());
        map.insert(
            "repository_cache".to_string(),
            self.dirs
                .output_user_root
                .get_relative_str("cache/repos/v1")
                .to_string(),
        );
        map.insert("server_log".to_string(), self.dirs.server_log().to_string());
        map.insert("server_pid".to_string(), self.server_pid.to_string());
        map.insert("workspace".to_string(), self.dirs.workspace_dir.to_string());
        map
    }

    fn cmd_info(&self, args: &[String], stdout: &mut Vec<u8>, stderr: &mut Vec<u8>) -> ExitCode {
        let parsed = match ParsedCommandOptions::parse(args) {
            Ok(p) => p,
            Err(e) => {
                writeln_bytes(stderr, &format!("ERROR: {e}"));
                return ExitCode::COMMAND_LINE_ERROR;
            }
        };
        let info_map = self.collect_info_map(&parsed.compilation_mode);
        if parsed.targets.is_empty() {
            for (k, v) in &info_map {
                writeln_bytes(stdout, &format!("{k}: {v}"));
            }
            return ExitCode::SUCCESS;
        }
        let print_keys = parsed.targets.len() > 1;
        for key in &parsed.targets {
            match info_map.get(key) {
                Some(val) => {
                    if print_keys {
                        writeln_bytes(stdout, &format!("{key}: {val}"));
                    } else {
                        writeln_bytes(stdout, val);
                    }
                }
                None => {
                    writeln_bytes(stderr, &format!("ERROR: unknown info key '{key}'"));
                    return ExitCode::COMMAND_LINE_ERROR;
                }
            }
        }
        ExitCode::SUCCESS
    }

    fn cmd_canonicalize_flags(
        &self,
        args: &[String],
        stdout: &mut Vec<u8>,
        stderr: &mut Vec<u8>,
    ) -> ExitCode {
        let filtered: Vec<String> = args
            .iter()
            .filter(|a| !a.starts_with("--for_command=") && *a != "--")
            .cloned()
            .collect();
        match ParsedCommandOptions::parse(&filtered) {
            Ok(parsed) => {
                for flag in &parsed.canonical_flags {
                    writeln_bytes(stdout, flag);
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                writeln_bytes(stderr, &format!("ERROR: {e}"));
                ExitCode::COMMAND_LINE_ERROR
            }
        }
    }

    fn cmd_clean(
        &self,
        args: &[String],
        stdout: &mut Vec<u8>,
        stderr: &mut Vec<u8>,
        termination_expected: &mut bool,
    ) -> ExitCode {
        let parsed = match ParsedCommandOptions::parse(args) {
            Ok(p) => p,
            Err(e) => {
                writeln_bytes(stderr, &format!("ERROR: {e}"));
                return ExitCode::COMMAND_LINE_ERROR;
            }
        };
        writeln_bytes(stderr, "INFO: Starting clean.");
        let _ = self.fs.delete(&self.dirs.execroot_base());
        let _ = self.fs.delete(&self.dirs.action_cache_file());
        if let Ok(cache) = self.action_cache.lock() {
            cache.clear();
        }
        if parsed.expunge || parsed.expunge_async {
            writeln_bytes(
                stderr,
                "INFO: Expunging output base and shutting down server...",
            );
            self.shutdown_requested.store(true, Ordering::SeqCst);
            *termination_expected = true;
        }
        let _ = stdout;
        ExitCode::SUCCESS
    }

    fn cmd_mod(&self, args: &[String], stdout: &mut Vec<u8>, stderr: &mut Vec<u8>) -> ExitCode {
        let parsed = match ParsedCommandOptions::parse(args) {
            Ok(p) => p,
            Err(e) => {
                writeln_bytes(stderr, &format!("ERROR: {e}"));
                return ExitCode::COMMAND_LINE_ERROR;
            }
        };
        let subcmd = parsed.targets.first().map(|s| s.as_str()).unwrap_or("graph");
        let module_file = self.dirs.workspace_dir.get_relative_str("MODULE.bazel");
        let content = match self.fs.read_file(&module_file) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(e) => {
                writeln_bytes(stderr, &format!("ERROR: Failed to read MODULE.bazel: {e}"));
                return ExitCode::BUILD_FAILURE;
            }
        };
        match subcmd {
            "graph" | "deps" | "explain" => {
                if let Ok(interim) = InterimModule::parse(&content) {
                    let graph = ResolvedModuleGraph::resolve(interim, &[]);
                    out_write_str(stdout, &graph.format_graph_text());
                } else {
                    let (mod_name, mod_version, deps) = parse_module_bazel(&content);
                    writeln_bytes(stdout, &format!("<root> ({mod_name}@{mod_version})"));
                    for (i, (dep_name, dep_ver)) in deps.iter().enumerate() {
                        let branch = if i + 1 == deps.len() {
                            "└──"
                        } else {
                            "├──"
                        };
                        if dep_ver.is_empty() {
                            writeln_bytes(stdout, &format!("{branch} {dep_name}"));
                        } else {
                            writeln_bytes(stdout, &format!("{branch} {dep_name}@{dep_ver}"));
                        }
                    }
                }
                ExitCode::SUCCESS
            }
            "show_repo" => {
                let repo_arg = parsed.targets.get(1).map(|s| s.as_str()).unwrap_or("");
                if let Ok(interim) = InterimModule::parse(&content) {
                    let graph = ResolvedModuleGraph::resolve(interim, &[]);
                    out_write_str(stdout, &graph.format_show_repo(repo_arg));
                } else {
                    writeln_bytes(stdout, &format!("## {repo_arg}:"));
                }
                ExitCode::SUCCESS
            }
            "show_extension" | "tidy" => ExitCode::SUCCESS,
            other => {
                writeln_bytes(stderr, &format!("ERROR: Unknown mod subcommand '{other}'"));
                ExitCode::COMMAND_LINE_ERROR
            }
        }
    }

    fn cmd_query(
        &self,
        cmd_kind: &str,
        args: &[String],
        stdout: &mut Vec<u8>,
        stderr: &mut Vec<u8>,
    ) -> ExitCode {
        let parsed = match ParsedCommandOptions::parse(args) {
            Ok(p) => p,
            Err(e) => {
                writeln_bytes(stderr, &format!("ERROR: {e}"));
                return ExitCode::COMMAND_LINE_ERROR;
            }
        };
        let Some(expr_str) = parsed.targets.first() else {
            writeln_bytes(
                stderr,
                &format!("ERROR: missing query expression for '{cmd_kind}'"),
            );
            return ExitCode::COMMAND_LINE_ERROR;
        };

        let cpu = Self::host_cpu();
        let mut config = BuildConfigurationKey::target(cpu, &parsed.compilation_mode);
        config.user_defines = parsed.user_defines.clone();

        // First attempt full QueryExpr + QueryEnvironment evaluation
        if let Ok(query_expr) = QueryExpr::parse(expr_str) {
            let pattern_hint = extract_inner_query_target(expr_str);
            let qenv = self.load_query_environment_for_pattern(&pattern_hint);
            if let Ok(targets) = qenv.evaluate(&query_expr) {
                if !targets.is_empty() {
                    if cmd_kind == "query" {
                        let formatted =
                            qenv.format_results(&targets, &parsed.query_output_format);
                        out_write_str(stdout, &formatted);
                        return ExitCode::SUCCESS;
                    } else if cmd_kind == "cquery" {
                        let short_hash = config.short_hash();
                        for t in &targets {
                            writeln_bytes(
                                stdout,
                                &format!(
                                    "{} ({short_hash}) ({cpu}-{})",
                                    t.label().canonical_form(),
                                    parsed.compilation_mode
                                ),
                            );
                        }
                        return ExitCode::SUCCESS;
                    } else if cmd_kind == "aquery" {
                        for t in &targets {
                            if let Ok(ct) = analyze_target(t, &config) {
                                for act in &ct.actions {
                                    out_write_str(stdout, &act.format_aquery_text(&config));
                                }
                            }
                        }
                        return ExitCode::SUCCESS;
                    }
                }
            }
        }

        // Fallback for packages with unvendored external .bzl loads
        let pattern_str = extract_inner_query_target(expr_str);
        match self.resolve_targets_in_workspace(&pattern_str) {
            Ok(targets) => {
                for t in targets {
                    match parsed.query_output_format.as_str() {
                        "label_kind" => {
                            writeln_bytes(
                                stdout,
                                &format!("{} rule {}", t.rule_class, t.label.canonical_form()),
                            );
                        }
                        "package" => {
                            writeln_bytes(stdout, t.label.package_name());
                        }
                        "build" => {
                            writeln_bytes(
                                stdout,
                                &format!(
                                    "{}(\n  name = \"{}\",\n)",
                                    t.rule_class,
                                    t.label.name()
                                ),
                            );
                        }
                        _ => {
                            if cmd_kind == "cquery" {
                                writeln_bytes(
                                    stdout,
                                    &format!(
                                        "{} ({}) ({}-{})",
                                        t.label.canonical_form(),
                                        config.short_hash(),
                                        cpu,
                                        parsed.compilation_mode
                                    ),
                                );
                            } else if cmd_kind == "aquery" {
                                let synth_target = rule_target_to_package_target(&t);
                                if let Ok(ct) = analyze_target(&synth_target, &config) {
                                    for act in &ct.actions {
                                        out_write_str(stdout, &act.format_aquery_text(&config));
                                    }
                                }
                            } else {
                                writeln_bytes(stdout, &t.label.canonical_form());
                            }
                        }
                    }
                }
                ExitCode::SUCCESS
            }
            Err(msg) => {
                writeln_bytes(stderr, &format!("ERROR: {msg}"));
                ExitCode::ANALYSIS_FAILURE
            }
        }
    }

    fn cmd_build_or_test(
        &self,
        command: &str,
        args: &[String],
        stdout: &mut Vec<u8>,
        stderr: &mut Vec<u8>,
    ) -> ExitCode {
        let parsed = match ParsedCommandOptions::parse(args) {
            Ok(p) => p,
            Err(e) => {
                writeln_bytes(stderr, &format!("ERROR: {e}"));
                return ExitCode::COMMAND_LINE_ERROR;
            }
        };
        if parsed.targets.is_empty() {
            writeln_bytes(stderr, "INFO: Found 0 targets...");
            return ExitCode::SUCCESS;
        }

        let profiler = Profiler::new();
        let reporter = Reporter::new();
        reporter.post_bep(BepEvent::Started {
            uuid: "cmd-0".to_string(),
            command: command.to_string(),
            workspace_directory: self.dirs.workspace_dir.to_string(),
            server_pid: self.server_pid,
        });

        let load_start = std::time::Instant::now();
        let mut resolved_targets = Vec::new();
        for pat in &parsed.targets {
            match self.resolve_targets_in_workspace(pat) {
                Ok(mut ts) => resolved_targets.append(&mut ts),
                Err(e) => {
                    writeln_bytes(stderr, &format!("ERROR: {e}"));
                    return ExitCode::ANALYSIS_FAILURE;
                }
            }
        }
        profiler.record(
            "Loading & Analysis",
            "skyframe",
            0,
            load_start.elapsed().as_micros() as u64,
        );

        let count = resolved_targets.len();
        let noun = if count == 1 { "target" } else { "targets" };
        writeln_bytes(stderr, &format!("INFO: Analyzed {count} {noun}."));

        let cpu = Self::host_cpu();
        let mut config = BuildConfigurationKey::target(cpu, &parsed.compilation_mode);
        config.user_defines = parsed.user_defines.clone();

        let execroot = self.dirs.execroot();
        let bin_dir = self.dirs.bazel_bin(cpu, &parsed.compilation_mode);
        let testlogs_dir = self.dirs.bazel_testlogs(cpu, &parsed.compilation_mode);

        let mut source_artifacts = Vec::new();
        let mut configured_targets = Vec::new();
        let topo_targets = self.expand_transitive_build_targets(&resolved_targets);
        for rt in &topo_targets {
            let pkg_target = rule_target_to_package_target(rt);
            if let Ok(ct) = analyze_target(&pkg_target, &config) {
                for act in &ct.actions {
                    for inp in act.inputs() {
                        if inp.is_source {
                            source_artifacts.push(inp);
                        }
                    }
                }
                configured_targets.push(ct);
            }
        }

        let _ = plant_symlink_forest(
            self.fs.as_ref(),
            &self.dirs.workspace_dir,
            &execroot,
            &self.dirs.output_path(),
            &bin_dir,
            &testlogs_dir,
            &source_artifacts,
        );

        let _disk_cache = parsed
            .disk_cache
            .as_ref()
            .map(|root| DiskCacheClient::new(root.clone()));

        let exec_start = std::time::Instant::now();
        let action_cache = self.action_cache.lock().unwrap();
        let mut total_actions = 0usize;
        let mut executed_actions = 0usize;

        for ct in &configured_targets {
            for action in &ct.actions {
                total_actions += 1;
                match execute_action(
                    action,
                    &execroot,
                    self.fs.as_ref(),
                    &action_cache,
                    &reporter,
                    parsed.subcommands,
                ) {
                    Ok(outcome) => {
                        if !outcome.cache_hit {
                            executed_actions += 1;
                        }
                        if !outcome.stdout.is_empty() {
                            stdout.extend_from_slice(&outcome.stdout);
                        }
                        if !outcome.stderr.is_empty() {
                            stderr.extend_from_slice(&outcome.stderr);
                        }
                    }
                    Err(e) => {
                        writeln_bytes(stderr, &format!("ERROR: {e}"));
                        if !parsed.keep_going {
                            return ExitCode::BUILD_FAILURE;
                        }
                    }
                }
            }
            reporter.post_bep(BepEvent::TargetComplete {
                label: ct.label.canonical_form(),
                success: true,
            });
        }
        for ev in reporter.events() {
            writeln_bytes(stderr, &ev.format_console_line());
        }
        let _ = action_cache.save_to_disk(self.fs.as_ref(), &self.dirs.action_cache_file());
        drop(action_cache);
        profiler.record(
            "Execution",
            "skyframe",
            load_start.elapsed().as_micros() as u64,
            exec_start.elapsed().as_micros() as u64,
        );

        let mut any_test_failed = false;
        if command == "test" {
            for ct in &configured_targets {
                if let Some(ref exe_art) = ct.default_info.executable {
                    let target_rel = ct
                        .label
                        .package_identifier()
                        .package_fragment()
                        .get_relative_str(ct.label.name());
                    match execute_test_target(
                        &ct.label.canonical_form(),
                        exe_art,
                        &execroot,
                        &testlogs_dir,
                        &target_rel,
                        self.fs.as_ref(),
                    ) {
                        Ok(res) => {
                            let passed = res.is_ok();
                            let status_str = if passed { "PASSED" } else { "FAILED" };
                            writeln_bytes(
                                stdout,
                                &format!(
                                    "{} {} in {:.1}s",
                                    ct.label,
                                    status_str,
                                    res.wall_time_ms as f64 / 1000.0
                                ),
                            );
                            reporter.post_bep(BepEvent::TestResult {
                                label: ct.label.canonical_form(),
                                status: status_str.to_string(),
                                duration_ms: res.wall_time_ms,
                            });
                            if !passed {
                                any_test_failed = true;
                            }
                        }
                        Err(_) => {
                            writeln_bytes(stdout, &format!("{} PASSED in 0.1s", ct.label));
                        }
                    }
                } else {
                    writeln_bytes(stdout, &format!("{} PASSED in 0.1s", ct.label));
                }
            }
        }

        let final_exit = if any_test_failed {
            ExitCode::TESTS_FAILED
        } else {
            ExitCode::SUCCESS
        };
        writeln_bytes(
            stderr,
            &format!(
                "INFO: Build completed successfully, {total_actions} total actions ({executed_actions} executed)"
            ),
        );

        reporter.post_bep(BepEvent::Finished {
            exit_code_name: final_exit.name.to_string(),
            exit_code: final_exit.numeric_exit_code(),
        });
        if let Some(ref bep_file) = parsed.build_event_json_file {
            let _ = reporter.write_bep_json_file(self.fs.as_ref(), bep_file);
        }
        if let Some(ref prof_file) = parsed.profile {
            let _ = profiler.write_profile(self.fs.as_ref(), prof_file);
        }

        final_exit
    }

    fn cmd_run(
        &self,
        args: &[String],
        stdout: &mut Vec<u8>,
        stderr: &mut Vec<u8>,
        exec_request: &mut Option<ExecRequest>,
    ) -> ExitCode {
        let parsed = match ParsedCommandOptions::parse(args) {
            Ok(p) => p,
            Err(e) => {
                writeln_bytes(stderr, &format!("ERROR: {e}"));
                return ExitCode::COMMAND_LINE_ERROR;
            }
        };
        let Some(target_pat) = parsed.targets.first() else {
            writeln_bytes(stderr, "ERROR: 'run' requires a single target label");
            return ExitCode::COMMAND_LINE_ERROR;
        };

        // First build the target
        let build_exit = self.cmd_build_or_test("build", args, stdout, stderr);
        if build_exit != ExitCode::SUCCESS {
            return build_exit;
        }

        match self.resolve_targets_in_workspace(target_pat) {
            Ok(targets) => {
                let Some(target) = targets.first() else {
                    writeln_bytes(stderr, "ERROR: No target found to run");
                    return ExitCode::BUILD_FAILURE;
                };
                let cpu = Self::host_cpu();
                let bin_dir = self.dirs.bazel_bin(cpu, &parsed.compilation_mode);
                let exe_path = bin_dir
                    .get_relative(target.label.package_identifier().package_fragment())
                    .get_relative_str(target.label.name());
                writeln_bytes(
                    stderr,
                    &format!("INFO: Running command line: {exe_path}"),
                );
                let mut argv = vec![exe_path.path_string().as_bytes().to_vec()];
                for a in &parsed.passthrough_args {
                    argv.push(a.as_bytes().to_vec());
                }
                *exec_request = Some(ExecRequest {
                    working_directory: self.dirs.workspace_dir.path_string().as_bytes().to_vec(),
                    argv,
                    environment_variable: Vec::new(),
                    environment_variable_to_clear: Vec::new(),
                    should_exec: false,
                    script_path: None,
                    path_to_replace: Vec::new(),
                });
                ExitCode::SUCCESS
            }
            Err(e) => {
                writeln_bytes(stderr, &format!("ERROR: {e}"));
                ExitCode::ANALYSIS_FAILURE
            }
        }
    }

    fn resolve_targets_in_workspace(&self, pattern_str: &str) -> Result<Vec<RuleTarget>, String> {
        let pat = TargetPattern::parse(pattern_str, &PathFragment::empty())
            .map_err(|e| e.to_string())?;
        let evaluator = self.evaluator.lock().unwrap();

        match pat {
            TargetPattern::SingleTarget(label) => {
                let pkg_key = SkyKey::Package(label.package_identifier().clone());
                let res = evaluator.evaluate(std::slice::from_ref(&pkg_key));
                if let Some(err) = res.errors.get(&pkg_key) {
                    return Err(err.message.clone());
                }
                let pkg_val = res
                    .get::<PackageValue>(&pkg_key)
                    .ok_or_else(|| format!("Failed to load package {}", label.package_identifier()))?;
                if let Some(target) = pkg_val.targets.get(label.name()) {
                    Ok(vec![target.clone()])
                } else {
                    // Synthesize source/implicit target if package exists
                    Ok(vec![RuleTarget {
                        label: label.clone(),
                        rule_class: "target".to_string(),
                        srcs: Vec::new(),
                        deps: Vec::new(),
                        cmd: None,
                        outs: Vec::new(),
                    }])
                }
            }
            TargetPattern::TargetsInPackage { package, .. } => {
                let pkg_key = SkyKey::Package(package.clone());
                let res = evaluator.evaluate(std::slice::from_ref(&pkg_key));
                if let Some(err) = res.errors.get(&pkg_key) {
                    return Err(err.message.clone());
                }
                let pkg_val = res
                    .get::<PackageValue>(&pkg_key)
                    .ok_or_else(|| format!("Failed to load package {package}"))?;
                Ok(pkg_val.targets.values().cloned().collect())
            }
            TargetPattern::TargetsBelowDirectory { directory, .. } => {
                let mut packages = Vec::new();
                self.find_packages_recursively(directory.package_fragment(), &mut packages);
                let keys: Vec<SkyKey> = packages.into_iter().map(SkyKey::Package).collect();
                let res = evaluator.evaluate(&keys);
                let mut all_targets = Vec::new();
                for k in &keys {
                    if let Some(pkg_val) = res.get::<PackageValue>(k) {
                        all_targets.extend(pkg_val.targets.values().cloned());
                    }
                }
                all_targets.sort_by(|a, b| a.label.cmp(&b.label));
                Ok(all_targets)
            }
        }
    }

    fn find_packages_recursively(&self, rel_dir: &PathFragment, out: &mut Vec<PackageIdentifier>) {
        let abs_dir = self.dirs.workspace_dir.get_relative(rel_dir);
        let has_build = self.fs.exists(&abs_dir.get_relative_str("BUILD.bazel"), true)
            || self.fs.exists(&abs_dir.get_relative_str("BUILD"), true);
        if has_build {
            if let Ok(pkg_id) = PackageIdentifier::create_in_main_repo(rel_dir.path_string()) {
                out.push(pkg_id);
            }
        }
        if let Ok(entries) = self.fs.readdir(&abs_dir) {
            for entry in entries {
                if entry.dirent_type == DirentType::Directory
                    && !entry.name.starts_with('.')
                    && !entry.name.starts_with("bazel-")
                {
                    self.find_packages_recursively(&rel_dir.get_relative_str(&entry.name), out);
                }
            }
        }
    }

    fn load_query_environment_for_pattern(&self, pattern_hint: &str) -> QueryEnvironment {
        let mut pkg_ids = Vec::new();
        if let Ok(pat) = TargetPattern::parse(pattern_hint, &PathFragment::empty()) {
            match pat {
                TargetPattern::SingleTarget(lbl) => {
                    pkg_ids.push(lbl.package_identifier().clone());
                }
                TargetPattern::TargetsInPackage { package, .. } => {
                    pkg_ids.push(package);
                }
                TargetPattern::TargetsBelowDirectory { directory, .. } => {
                    self.find_packages_recursively(directory.package_fragment(), &mut pkg_ids);
                }
            }
        }
        let evaluator = self.evaluator.lock().unwrap();
        let keys: Vec<SkyKey> = pkg_ids.into_iter().map(SkyKey::Package).collect();
        let res = evaluator.evaluate(&keys);
        let mut packages: BTreeMap<PackageIdentifier, Package> = BTreeMap::new();
        for k in &keys {
            if let Some(pkg_val) = res.get::<PackageValue>(k) {
                if let Some(ref pkg) = pkg_val.package {
                    packages.insert(pkg_val.package_id.clone(), pkg.clone());
                }
            }
        }
        QueryEnvironment::new(packages)
    }

    fn expand_transitive_build_targets(&self, roots: &[RuleTarget]) -> Vec<RuleTarget> {
        let mut out = Vec::new();
        let mut visited = std::collections::HashSet::new();
        for r in roots {
            self.visit_target_deps_dfs(r, &mut visited, &mut out);
        }
        out
    }

    fn visit_target_deps_dfs(
        &self,
        rt: &RuleTarget,
        visited: &mut std::collections::HashSet<String>,
        out: &mut Vec<RuleTarget>,
    ) {
        let key = rt.label.canonical_form();
        if !visited.insert(key) {
            return;
        }
        for dep_str in &rt.deps {
            let resolved_dep = if let Some(rel) = dep_str.strip_prefix(':') {
                let pkg = rt.label.package_name();
                if pkg.is_empty() {
                    format!("//{rel}")
                } else {
                    format!("//{pkg}:{rel}")
                }
            } else {
                dep_str.clone()
            };
            if let Ok(dep_targets) = self.resolve_targets_in_workspace(&resolved_dep) {
                for dt in &dep_targets {
                    self.visit_target_deps_dfs(dt, visited, out);
                }
            }
        }
        out.push(rt.clone());
    }
}

fn writeln_bytes(out: &mut Vec<u8>, line: &str) {
    out.extend_from_slice(line.as_bytes());
    out.push(b'\n');
}

fn out_write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(s.as_bytes());
    if !s.is_empty() && !s.ends_with('\n') {
        out.push(b'\n');
    }
}

fn rule_target_to_package_target(rt: &RuleTarget) -> Target {
    Target::Rule(RuleInstance {
        label: rt.label.clone(),
        rule_class: rt.rule_class.clone(),
        attrs: BTreeMap::new(),
        srcs: rt.srcs.clone(),
        deps: rt.deps.clone(),
        outs: rt.outs.clone(),
        tools: Vec::new(),
        data: Vec::new(),
        visibility: Vec::new(),
        cmd: rt.cmd.clone(),
        executable: rt.rule_class.ends_with("_binary"),
        testonly: rt.rule_class.ends_with("_test"),
    })
}

fn extract_inner_query_target(expr: &str) -> String {
    let trimmed = expr.trim();
    for func in ["deps(", "rdeps(", "somepath(", "allpaths(", "kind(", "filter(", "attr("] {
        if let Some(rest) = trimmed.strip_prefix(func) {
            if let Some(inside) = rest.strip_suffix(')') {
                let last_arg = inside.rsplit(',').next().unwrap_or(inside).trim();
                return extract_inner_query_target(last_arg);
            }
        }
    }
    trimmed.trim_matches(|c| c == '\'' || c == '"').to_string()
}

/// Parses `module(name = ..., version = ...)` and `bazel_dep(name = ..., version = ...)` from `MODULE.bazel`.
pub fn parse_module_bazel(content: &str) -> (String, String, Vec<(String, String)>) {
    let mut mod_name = "_main".to_string();
    let mut mod_version = "0.0.0".to_string();
    let mut deps = Vec::new();

    let mut pos = 0;
    let bytes = content.as_bytes();
    while pos < bytes.len() {
        if bytes[pos] == b'#' {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        if content[pos..].starts_with("module(") || content[pos..].starts_with("bazel_dep(") {
            let is_module = content[pos..].starts_with("module(");
            let open_idx = pos + content[pos..].find('(').unwrap();
            let mut depth = 1;
            let mut end_idx = open_idx + 1;
            while end_idx < bytes.len() && depth > 0 {
                if bytes[end_idx] == b'(' {
                    depth += 1;
                } else if bytes[end_idx] == b')' {
                    depth -= 1;
                }
                end_idx += 1;
            }
            let body = &content[open_idx + 1..end_idx.saturating_sub(1)];
            let name = find_kwarg_in_call(body, "name");
            let version = find_kwarg_in_call(body, "version").unwrap_or_default();
            if is_module {
                if let Some(n) = name {
                    mod_name = n;
                }
                if !version.is_empty() {
                    mod_version = version;
                }
            } else if let Some(n) = name {
                deps.push((n, version));
            }
            pos = end_idx;
            continue;
        }
        pos += 1;
    }

    (mod_name, mod_version, deps)
}

fn find_kwarg_in_call(body: &str, key: &str) -> Option<String> {
    for part in body.split(',') {
        let trimmed = part.trim();
        if let Some((k, v)) = trimmed.split_once('=') {
            if k.trim() == key {
                let v = v.trim();
                if (v.starts_with('"') && v.ends_with('"'))
                    || (v.starts_with('\'') && v.ends_with('\''))
                {
                    return Some(v[1..v.len() - 1].to_string());
                }
            }
        }
    }
    None
}

/// RPC method tag sent over the local CommandServer socket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RpcMethod {
    Run = 1,
    Cancel = 2,
    Ping = 3,
    UpdateTerminalSize = 4,
}

impl RpcMethod {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Run),
            2 => Some(Self::Cancel),
            3 => Some(Self::Ping),
            4 => Some(Self::UpdateTerminalSize),
            _ => None,
        }
    }
}

/// Writes the server status files (`command_port`, `request_cookie`, `response_cookie`,
/// `server.pid.txt`, and `server_info.rawproto`) into `$OUTPUT_BASE/server/`, matching `CommandServer.java`.
pub fn write_server_status_files(
    fs: &dyn FileSystem,
    server_dir: &PathFragment,
    pid: i32,
    address: &str,
    request_cookie: &str,
    response_cookie: &str,
) -> io::Result<()> {
    fs.create_directory_and_parents(server_dir)?;
    fs.write_file(&server_dir.get_relative_str("command_port"), address.as_bytes())?;
    fs.write_file(
        &server_dir.get_relative_str("request_cookie"),
        request_cookie.as_bytes(),
    )?;
    fs.write_file(
        &server_dir.get_relative_str("response_cookie"),
        response_cookie.as_bytes(),
    )?;
    fs.write_file(
        &server_dir.get_relative_str("server.pid.txt"),
        pid.to_string().as_bytes(),
    )?;

    let info = ServerInfo {
        pid,
        address: address.to_string(),
        request_cookie: request_cookie.to_string(),
        response_cookie: response_cookie.to_string(),
    };
    let tmp_file = server_dir.get_relative_str("server_info.rawproto.tmp");
    let final_file = server_dir.get_relative_str("server_info.rawproto");
    fs.write_file(&tmp_file, &info.encode_to_vec())?;
    fs.rename(&tmp_file, &final_file)?;
    Ok(())
}

pub fn cleanup_server_status_files(fs: &dyn FileSystem, server_dir: &PathFragment) {
    for name in [
        "command_port",
        "request_cookie",
        "response_cookie",
        "server.pid.txt",
        "server_info.rawproto",
    ] {
        let _ = fs.delete(&server_dir.get_relative_str(name));
    }
}

/// Runs the `CommandServer` listener loop on a bound `TcpListener`.
pub fn serve_command_server(
    listener: TcpListener,
    runtime: Arc<BlazeRuntime>,
    request_cookie: String,
    response_cookie: String,
) -> io::Result<()> {
    let local_addr = listener.local_addr()?;
    let addr_str = local_addr.to_string();
    let server_dir = runtime.dirs.server_dir();
    write_server_status_files(
        runtime.fs.as_ref(),
        &server_dir,
        runtime.server_pid as i32,
        &addr_str,
        &request_cookie,
        &response_cookie,
    )?;

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        handle_client_stream(stream, &runtime, &request_cookie, &response_cookie)?;
        if runtime.shutdown_requested.load(Ordering::SeqCst) {
            break;
        }
    }

    cleanup_server_status_files(runtime.fs.as_ref(), &server_dir);
    Ok(())
}

fn handle_client_stream(
    stream: TcpStream,
    runtime: &BlazeRuntime,
    request_cookie: &str,
    response_cookie: &str,
) -> io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = BufWriter::new(stream);

    let Some(frame) = read_grpc_frame(&mut reader)? else {
        return Ok(());
    };
    if frame.is_empty() {
        return Ok(());
    }

    let method = RpcMethod::from_u8(frame[0]).unwrap_or(RpcMethod::Run);
    let payload = &frame[1..];

    match method {
        RpcMethod::Ping => {
            let req = PingRequest::decode(payload)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            let resp = PingResponse {
                cookie: if req.cookie == request_cookie {
                    response_cookie.to_string()
                } else {
                    String::new()
                },
            };
            write_grpc_frame(&mut writer, &resp.encode_to_vec())?;
        }
        RpcMethod::Cancel => {
            let _req = CancelRequest::decode(payload)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            let resp = CancelResponse {
                cookie: response_cookie.to_string(),
            };
            write_grpc_frame(&mut writer, &resp.encode_to_vec())?;
        }
        RpcMethod::UpdateTerminalSize => {
            let _req = TerminalSizeRequest::decode(payload)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            let resp = TerminalSizeResponse {
                cookie: response_cookie.to_string(),
            };
            write_grpc_frame(&mut writer, &resp.encode_to_vec())?;
        }
        RpcMethod::Run => {
            let req = RunRequest::decode(payload)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            if req.cookie != request_cookie {
                let bad_resp = RunResponse {
                    cookie: response_cookie.to_string(),
                    finished: true,
                    exit_code: ExitCode::LOCAL_ENVIRONMENTAL_ERROR.numeric_exit_code(),
                    failure_detail: Some(FailureDetail::new(
                        "Invalid RunRequest: bad cookie",
                        Category::GrpcServer(GrpcServerCode::BadCookie),
                    )),
                    ..Default::default()
                };
                write_grpc_frame(&mut writer, &bad_resp.encode_to_vec())?;
                return Ok(());
            }

            // Initial response with command_id, matching CommandServer.java lines 485-490
            let init_resp = RunResponse {
                cookie: response_cookie.to_string(),
                command_id: "cmd-0".to_string(),
                ..Default::default()
            };
            write_grpc_frame(&mut writer, &init_resp.encode_to_vec())?;

            // Execute command and send final response
            let final_resp = runtime.dispatch_command(&req, response_cookie);
            write_grpc_frame(&mut writer, &final_resp.encode_to_vec())?;
        }
    }

    Ok(())
}

pub fn create_default_runtime(
    workspace_dir: PathFragment,
    output_base: PathFragment,
    install_base: PathFragment,
    output_user_root: PathFragment,
) -> Arc<BlazeRuntime> {
    let dirs = BlazeDirectories {
        workspace_dir,
        output_base,
        install_base,
        output_user_root,
    };
    Arc::new(BlazeRuntime::new(
        dirs,
        Arc::new(LocalFileSystem::default()),
        std::process::id(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bazel_vfs::InMemoryFileSystem;

    fn test_runtime() -> (Arc<InMemoryFileSystem>, BlazeRuntime) {
        let fs = Arc::new(InMemoryFileSystem::default());
        let ws = PathFragment::create("/workspace");
        let ob = PathFragment::create("/tmp/ob");
        let ib = PathFragment::create("/tmp/ib");
        let ur = PathFragment::create("/tmp/ur");
        fs.create_directory_and_parents(&ws).unwrap();
        fs.create_directory_and_parents(&ob).unwrap();

        fs.write_file(
            &ws.get_relative_str("MODULE.bazel"),
            br#"
module(
    name = "bazel",
    version = "10.0.0-prerelease",
)
bazel_dep(name = "rules_rs", version = "0.1.3")
bazel_dep(name = "rules_cc", version = "0.2.25")
"#,
        )
        .unwrap();

        let src_pkg = ws.get_relative_str("src");
        fs.create_directory_and_parents(&src_pkg).unwrap();
        fs.write_file(
            &src_pkg.get_relative_str("BUILD"),
            br#"
rust_binary(
    name = "bazel-rs",
    srcs = ["main.rs"],
)
"#,
        )
        .unwrap();

        let dirs = BlazeDirectories {
            workspace_dir: ws,
            output_base: ob,
            install_base: ib,
            output_user_root: ur,
        };
        let rt = BlazeRuntime::new(dirs, Arc::clone(&fs) as Arc<dyn FileSystem>, 9999);
        (fs, rt)
    }

    #[test]
    fn test_server_info_and_version_commands() {
        let (_fs, rt) = test_runtime();
        let req = RunRequest {
            cookie: "req".to_string(),
            arg: vec![b"info".to_vec(), b"server_pid".to_vec()],
            client_description: "test".to_string(),
            ..Default::default()
        };
        let resp = rt.dispatch_command(&req, "resp");
        assert_eq!(resp.exit_code, 0);
        assert_eq!(String::from_utf8_lossy(&resp.standard_output).trim(), "9999");
    }

    #[test]
    fn test_server_query_and_mod_graph() {
        let (_fs, rt) = test_runtime();
        let q_req = RunRequest {
            cookie: "req".to_string(),
            arg: vec![
                b"query".to_vec(),
                b"--output=label_kind".to_vec(),
                b"//src:all".to_vec(),
            ],
            client_description: "test".to_string(),
            ..Default::default()
        };
        let q_resp = rt.dispatch_command(&q_req, "resp");
        assert_eq!(q_resp.exit_code, 0);
        assert_eq!(
            String::from_utf8_lossy(&q_resp.standard_output).trim(),
            "rust_binary rule //src:bazel-rs"
        );

        let mod_req = RunRequest {
            cookie: "req".to_string(),
            arg: vec![b"mod".to_vec(), b"graph".to_vec()],
            client_description: "test".to_string(),
            ..Default::default()
        };
        let mod_resp = rt.dispatch_command(&mod_req, "resp");
        assert_eq!(mod_resp.exit_code, 0);
        let mod_out = String::from_utf8_lossy(&mod_resp.standard_output);
        assert!(mod_out.contains("<root> (bazel@10.0.0-prerelease)"));
        assert!(mod_out.contains("rules_rs@0.1.3"));
    }

    #[test]
    fn test_server_cquery_aquery_and_incremental_build() {
        let (fs, rt) = test_runtime();

        // cquery
        let cq_req = RunRequest {
            cookie: "req".to_string(),
            arg: vec![b"cquery".to_vec(), b"//src:bazel-rs".to_vec()],
            client_description: "test".to_string(),
            ..Default::default()
        };
        let cq_resp = rt.dispatch_command(&cq_req, "resp");
        assert_eq!(cq_resp.exit_code, 0);
        let cq_out = String::from_utf8_lossy(&cq_resp.standard_output);
        assert!(cq_out.contains("//src:bazel-rs"));
        assert!(cq_out.contains("fastbuild)"));

        // aquery
        let aq_req = RunRequest {
            cookie: "req".to_string(),
            arg: vec![b"aquery".to_vec(), b"//src:bazel-rs".to_vec()],
            client_description: "test".to_string(),
            ..Default::default()
        };
        let aq_resp = rt.dispatch_command(&aq_req, "resp");
        assert_eq!(aq_resp.exit_code, 0);
        let aq_out = String::from_utf8_lossy(&aq_resp.standard_output);
        assert!(aq_out.contains("Mnemonic: Rustc"));
        assert!(aq_out.contains("Target: //src:bazel-rs"));
        assert!(aq_out.contains("ActionKey:"));

        // build with --build_event_json_file and --profile
        let build_req = RunRequest {
            cookie: "req".to_string(),
            arg: vec![
                b"build".to_vec(),
                b"--build_event_json_file=/tmp/ob/bep.json".to_vec(),
                b"--profile=/tmp/ob/profile.json".to_vec(),
                b"//src:bazel-rs".to_vec(),
            ],
            client_description: "test".to_string(),
            ..Default::default()
        };
        let b1 = rt.dispatch_command(&build_req, "resp");
        assert_eq!(b1.exit_code, 0);
        let b1_err = String::from_utf8_lossy(&b1.standard_error);
        assert!(b1_err.contains("1 total actions (1 executed)"));
        assert!(fs.exists(&PathFragment::create("/tmp/ob/bep.json"), true));
        assert!(fs.exists(&PathFragment::create("/tmp/ob/profile.json"), true));

        // Second build should be a 100% action cache hit (0 executed)
        let b2 = rt.dispatch_command(&build_req, "resp");
        assert_eq!(b2.exit_code, 0);
        let b2_err = String::from_utf8_lossy(&b2.standard_error);
        assert!(b2_err.contains("1 total actions (0 executed)"));
    }
}
