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

//! Bazel startup & command-line option parser and `.bazelrc` processor.
//! Mirrors `src/main/cpp/option_processor.cc` and `src/main/java/com/google/devtools/common/options`.

use bazel_proto::command_server::StartupOption as ProtoStartupOption;
use bazel_util::{md5, Os};
use bazel_vfs::{FileSystem, PathFragment};
use std::collections::{BTreeMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionsParsingError(pub String);

impl fmt::Display for OptionsParsingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for OptionsParsingError {}

/// Known nullary startup flags (do not take a separate next-arg value).
const NULLARY_STARTUP_FLAGS: &[&str] = &[
    "--batch",
    "--nobatch",
    "--block_for_lock",
    "--noblock_for_lock",
    "--quiet",
    "--noquiet",
    "--ignore_all_rc_files",
    "--noignore_all_rc_files",
    "--System_rc",
    "--nosystem_rc",
    "--system_rc",
    "--workspace_rc",
    "--noworkspace_rc",
    "--home_rc",
    "--nohome_rc",
    "--autodetect_server_javabase",
    "--noautodetect_server_javabase",
    "--Shutdown_on_low_sys_mem",
    "--shutdown_on_low_sys_mem",
    "--noshutdown_on_low_sys_mem",
    "--idle_server_tasks",
    "--noidle_server_tasks",
    "--write_command_log",
    "--nowrite_command_log",
    "--client_debug",
    "--noclient_debug",
    "--preemptible",
    "--nopreemptible",
];

/// Known unary startup flags (can be passed as `--flag=val` or `--flag val`).
const UNARY_STARTUP_FLAGS: &[&str] = &[
    "--output_base",
    "--output_user_root",
    "--install_base",
    "--max_idle_secs",
    "--bazelrc",
    "--host_jvm_args",
    "--server_javabase",
    "--failure_detail_out",
    "--connect_timeout_secs",
    "--local_startup_timeout_secs",
    "--digest_function",
    "--unix_digest_hash_attribute_name",
];

/// Parsed startup options affecting the client and server lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupOptions {
    pub output_base: Option<PathFragment>,
    pub output_user_root: Option<PathFragment>,
    pub install_base: Option<PathFragment>,
    pub batch: bool,
    pub block_for_lock: bool,
    pub quiet: bool,
    pub max_idle_secs: u64,
    pub bazelrc_files: Vec<String>,
    pub ignore_all_rc_files: bool,
    pub system_rc: bool,
    pub workspace_rc: bool,
    pub home_rc: bool,
    pub host_jvm_args: Vec<String>,
    pub raw_tagged_options: Vec<ProtoStartupOption>,
}

impl Default for StartupOptions {
    fn default() -> Self {
        Self {
            output_base: None,
            output_user_root: None,
            install_base: None,
            batch: false,
            block_for_lock: true,
            quiet: false,
            max_idle_secs: 3 * 3600,
            bazelrc_files: Vec::new(),
            ignore_all_rc_files: false,
            system_rc: true,
            workspace_rc: true,
            home_rc: true,
            host_jvm_args: Vec::new(),
            raw_tagged_options: Vec::new(),
        }
    }
}

impl StartupOptions {
    /// Computes the default output user root (`$HOME/.cache/bazel/_bazel_$USER` on Linux,
    /// `/var/tmp/_bazel_$USER` on macOS).
    pub fn resolve_output_user_root(&self, user: &str, home: Option<&str>) -> PathFragment {
        if let Some(ref root) = self.output_user_root {
            return root.clone();
        }
        match Os::current() {
            Os::Darwin => PathFragment::create(&format!("/var/tmp/_bazel_{user}")),
            _ => {
                if let Some(h) = home {
                    PathFragment::create(&format!("{h}/.cache/bazel/_bazel_{user}"))
                } else {
                    PathFragment::create(&format!("/tmp/_bazel_{user}"))
                }
            }
        }
    }

    /// Computes the workspace-specific output base (`$OUTPUT_USER_ROOT/<md5(workspace_dir)>`)
    /// unless overridden by `--output_base`.
    pub fn resolve_output_base(
        &self,
        workspace_dir: &PathFragment,
        user: &str,
        home: Option<&str>,
    ) -> PathFragment {
        if let Some(ref ob) = self.output_base {
            return ob.clone();
        }
        let user_root = self.resolve_output_user_root(user, home);
        let ws_hash = md5::hex_digest(workspace_dir.path_string().as_bytes());
        user_root.get_relative_str(&ws_hash)
    }

    pub fn resolve_install_base(
        &self,
        install_md5: &str,
        user: &str,
        home: Option<&str>,
    ) -> PathFragment {
        if let Some(ref ib) = self.install_base {
            return ib.clone();
        }
        let user_root = self.resolve_output_user_root(user, home);
        user_root
            .get_relative_str("install")
            .get_relative_str(install_md5)
    }

    pub fn apply_flag(&mut self, flag: &str, source: &str) -> Result<(), OptionsParsingError> {
        self.raw_tagged_options.push(ProtoStartupOption {
            option: flag.as_bytes().to_vec(),
            source: source.as_bytes().to_vec(),
        });

        if flag == "--batch" {
            self.batch = true;
        } else if flag == "--nobatch" {
            self.batch = false;
        } else if flag == "--block_for_lock" {
            self.block_for_lock = true;
        } else if flag == "--noblock_for_lock" {
            self.block_for_lock = false;
        } else if flag == "--quiet" {
            self.quiet = true;
        } else if flag == "--noquiet" {
            self.quiet = false;
        } else if flag == "--ignore_all_rc_files" {
            self.ignore_all_rc_files = true;
        } else if flag == "--noignore_all_rc_files" {
            self.ignore_all_rc_files = false;
        } else if flag == "--nosystem_rc" {
            self.system_rc = false;
        } else if flag == "--system_rc" {
            self.system_rc = true;
        } else if flag == "--noworkspace_rc" {
            self.workspace_rc = false;
        } else if flag == "--workspace_rc" {
            self.workspace_rc = true;
        } else if flag == "--nohome_rc" {
            self.home_rc = false;
        } else if flag == "--home_rc" {
            self.home_rc = true;
        } else if let Some(val) = flag.strip_prefix("--output_base=") {
            self.output_base = Some(PathFragment::create(val));
        } else if let Some(val) = flag.strip_prefix("--output_user_root=") {
            self.output_user_root = Some(PathFragment::create(val));
        } else if let Some(val) = flag.strip_prefix("--install_base=") {
            self.install_base = Some(PathFragment::create(val));
        } else if let Some(val) = flag.strip_prefix("--max_idle_secs=") {
            self.max_idle_secs = val.parse::<u64>().map_err(|_| {
                OptionsParsingError(format!("Invalid --max_idle_secs value: '{val}'"))
            })?;
        } else if let Some(val) = flag.strip_prefix("--bazelrc=") {
            self.bazelrc_files.push(val.to_string());
        } else if let Some(val) = flag.strip_prefix("--host_jvm_args=") {
            self.host_jvm_args.push(val.to_string());
        } else if NULLARY_STARTUP_FLAGS.contains(&flag)
            || UNARY_STARTUP_FLAGS
                .iter()
                .any(|p| flag.starts_with(&format!("{p}=")))
        {
            // Accepted startup flag
        } else {
            return Err(OptionsParsingError(format!(
                "Unknown startup option: '{flag}'.\n  For more info, run 'bazel help startup_options'."
            )));
        }
        Ok(())
    }
}

/// Splits raw CLI argv (excluding argv[0]) into `(startup_options, command, command_args)`.
pub fn split_command_line(
    args: &[String],
) -> Result<(StartupOptions, String, Vec<String>), OptionsParsingError> {
    let mut startup = StartupOptions::default();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if !arg.starts_with('-') {
            let command = arg.clone();
            let command_args = args[i + 1..].to_vec();
            return Ok((startup, command, command_args));
        }
        if arg == "--help" || arg == "-h" {
            return Ok((startup, "help".to_string(), Vec::new()));
        }
        if arg == "--version" {
            return Ok((
                startup,
                "version".to_string(),
                vec!["--startup_version".to_string()],
            ));
        }
        if UNARY_STARTUP_FLAGS.contains(&arg.as_str()) {
            if i + 1 >= args.len() {
                return Err(OptionsParsingError(format!(
                    "Startup option '{arg}' expects a value."
                )));
            }
            let combined = format!("{}={}", arg, args[i + 1]);
            startup.apply_flag(&combined, "")?;
            i += 2;
        } else {
            startup.apply_flag(arg, "")?;
            i += 1;
        }
    }
    // No command provided: defaults to "help"
    Ok((startup, String::new(), Vec::new()))
}

/// Parsed `.bazelrc` file containing command and config entries (`build`, `common`, `build:remote`, etc.).
#[derive(Debug, Clone, Default)]
pub struct BazelrcConfig {
    /// Maps section key (e.g., `"startup"`, `"common"`, `"build"`, `"build:opt"`) to list of `(option, source_path)`.
    pub sections: BTreeMap<String, Vec<(String, String)>>,
}

impl BazelrcConfig {
    pub fn parse_str(
        &mut self,
        content: &str,
        source_path: &str,
        workspace_dir: &PathFragment,
        fs: &dyn FileSystem,
        visited: &mut HashSet<PathFragment>,
    ) -> Result<(), OptionsParsingError> {
        let mut continued_line = String::new();
        for raw_line in content.lines() {
            let trimmed_comment = match raw_line.find('#') {
                Some(idx) => &raw_line[..idx],
                None => raw_line,
            };
            let trimmed = trimmed_comment.trim_end();
            if let Some(without_slash) = trimmed.strip_suffix('\\') {
                continued_line.push_str(without_slash);
                continued_line.push(' ');
                continue;
            }
            let full_line = if continued_line.is_empty() {
                trimmed.trim().to_string()
            } else {
                let mut s = std::mem::take(&mut continued_line);
                s.push_str(trimmed.trim());
                s
            };
            if full_line.is_empty() {
                continue;
            }

            let tokens = tokenize_rc_line(&full_line);
            if tokens.is_empty() {
                continue;
            }

            let first = &tokens[0];
            if first == "import" || first == "try-import" {
                let is_try = first == "try-import";
                if tokens.len() < 2 {
                    return Err(OptionsParsingError(format!(
                        "Invalid {first} statement in {source_path}"
                    )));
                }
                let raw_import_path =
                    tokens[1].replace("%workspace%", workspace_dir.path_string());
                let import_pf = PathFragment::create(&raw_import_path);
                let resolved_pf = if import_pf.is_absolute() {
                    import_pf
                } else {
                    workspace_dir.get_relative(&import_pf)
                };
                if visited.contains(&resolved_pf) {
                    continue;
                }
                match fs.read_file(&resolved_pf) {
                    Ok(bytes) => {
                        visited.insert(resolved_pf.clone());
                        let sub_content = String::from_utf8_lossy(&bytes);
                        self.parse_str(
                            &sub_content,
                            resolved_pf.path_string(),
                            workspace_dir,
                            fs,
                            visited,
                        )?;
                    }
                    Err(e) => {
                        if !is_try {
                            return Err(OptionsParsingError(format!(
                                "Cannot import '{}': {}",
                                resolved_pf, e
                            )));
                        }
                    }
                }
                continue;
            }

            let section = first.clone();
            let entry = self.sections.entry(section).or_default();
            for tok in &tokens[1..] {
                entry.push((tok.clone(), source_path.to_string()));
            }
        }
        Ok(())
    }

    /// Expands command options for a given `command` (including `always`, `common`, `<command>`, and `--config=<name>`).
    pub fn expand_command_args(
        &self,
        command: &str,
        cli_args: &[String],
    ) -> Result<Vec<String>, OptionsParsingError> {
        let mut base_args = Vec::new();
        for prefix in ["always", "common", command] {
            if let Some(opts) = self.sections.get(prefix) {
                for (opt, _) in opts {
                    base_args.push(opt.clone());
                }
            }
        }
        base_args.extend(cli_args.iter().cloned());

        let mut expanded = Vec::new();
        let mut seen_configs = HashSet::new();
        self.expand_configs_recursive(command, &base_args, &mut expanded, &mut seen_configs)?;
        Ok(expanded)
    }

    fn expand_configs_recursive(
        &self,
        command: &str,
        args: &[String],
        out: &mut Vec<String>,
        seen_configs: &mut HashSet<String>,
    ) -> Result<(), OptionsParsingError> {
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            if arg == "--" {
                out.extend_from_slice(&args[i..]);
                break;
            }
            let config_name = if let Some(c) = arg.strip_prefix("--config=") {
                Some((c.to_string(), 1))
            } else if arg == "--config" && i + 1 < args.len() {
                Some((args[i + 1].clone(), 2))
            } else {
                None
            };

            if let Some((cfg, consumed)) = config_name {
                if seen_configs.insert(cfg.clone()) {
                    let mut cfg_opts = Vec::new();
                    let common_key = format!("common:{cfg}");
                    let cmd_key = format!("{command}:{cfg}");
                    let mut found = false;
                    if let Some(opts) = self.sections.get(&common_key) {
                        found = true;
                        cfg_opts.extend(opts.iter().map(|(o, _)| o.clone()));
                    }
                    if let Some(opts) = self.sections.get(&cmd_key) {
                        found = true;
                        cfg_opts.extend(opts.iter().map(|(o, _)| o.clone()));
                    }
                    if !found {
                        return Err(OptionsParsingError(format!(
                            "Config value '{cfg}' is not defined in any .rc file"
                        )));
                    }
                    self.expand_configs_recursive(command, &cfg_opts, out, seen_configs)?;
                }
                i += consumed;
            } else {
                out.push(arg.clone());
                i += 1;
            }
        }
        Ok(())
    }
}

pub fn tokenize_rc_line(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' && !in_single {
            if let Some(next_c) = chars.next() {
                current.push(next_c);
            }
            continue;
        }
        if c == '\'' && !in_double {
            in_single = !in_single;
            continue;
        }
        if c == '"' && !in_single {
            in_double = !in_double;
            continue;
        }
        if c.is_whitespace() && !in_single && !in_double {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            continue;
        }
        current.push(c);
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Parsed command options for `build`, `test`, `query`, `cquery`, `aquery`, `clean`, `info`, etc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedCommandOptions {
    pub compilation_mode: String,
    pub jobs: usize,
    pub keep_going: bool,
    pub subcommands: bool,
    pub verbose_failures: bool,
    pub expunge: bool,
    pub expunge_async: bool,
    pub query_output_format: String,
    pub show_make_env: bool,
    pub disk_cache: Option<PathFragment>,
    pub profile: Option<PathFragment>,
    pub build_event_json_file: Option<PathFragment>,
    pub test_output: String,
    pub test_filter: Option<String>,
    pub spawn_strategy: String,
    pub user_defines: BTreeMap<String, String>,
    pub canonical_flags: Vec<String>,
    pub targets: Vec<String>,
    pub passthrough_args: Vec<String>,
}

impl Default for ParsedCommandOptions {
    fn default() -> Self {
        Self {
            compilation_mode: "fastbuild".to_string(),
            jobs: 8,
            keep_going: false,
            subcommands: false,
            verbose_failures: false,
            expunge: false,
            expunge_async: false,
            query_output_format: "label".to_string(),
            show_make_env: false,
            disk_cache: None,
            profile: None,
            build_event_json_file: None,
            test_output: "summary".to_string(),
            test_filter: None,
            spawn_strategy: "local".to_string(),
            user_defines: BTreeMap::new(),
            canonical_flags: Vec::new(),
            targets: Vec::new(),
            passthrough_args: Vec::new(),
        }
    }
}

impl ParsedCommandOptions {
    pub fn parse(args: &[String]) -> Result<Self, OptionsParsingError> {
        let mut opts = Self::default();
        let mut flag_map: BTreeMap<String, String> = BTreeMap::new();
        let mut i = 0;

        while i < args.len() {
            let arg = &args[i];
            if arg == "--" {
                opts.passthrough_args.extend_from_slice(&args[i + 1..]);
                break;
            }
            if !arg.starts_with('-') || arg == "-" {
                opts.targets.push(arg.clone());
                i += 1;
                continue;
            }

            if arg == "-c" || arg == "--compilation_mode" {
                if i + 1 >= args.len() {
                    return Err(OptionsParsingError(format!(
                        "Option '{arg}' expects a value"
                    )));
                }
                let val = &args[i + 1];
                Self::validate_compilation_mode(val)?;
                opts.compilation_mode = val.clone();
                flag_map.insert(
                    "compilation_mode".to_string(),
                    format!("--compilation_mode={val}"),
                );
                i += 2;
            } else if let Some(val) = arg.strip_prefix("--compilation_mode=") {
                Self::validate_compilation_mode(val)?;
                opts.compilation_mode = val.to_string();
                flag_map.insert(
                    "compilation_mode".to_string(),
                    format!("--compilation_mode={val}"),
                );
                i += 1;
            } else if arg == "-k" || arg == "--keep_going" {
                opts.keep_going = true;
                flag_map.insert("keep_going".to_string(), "--keep_going=1".to_string());
                i += 1;
            } else if arg == "--nokeep_going" {
                opts.keep_going = false;
                flag_map.insert("keep_going".to_string(), "--keep_going=0".to_string());
                i += 1;
            } else if arg == "-s" || arg == "--subcommands" {
                opts.subcommands = true;
                flag_map.insert("subcommands".to_string(), "--subcommands=true".to_string());
                i += 1;
            } else if arg == "--nosubcommands" {
                opts.subcommands = false;
                flag_map.insert("subcommands".to_string(), "--subcommands=false".to_string());
                i += 1;
            } else if arg == "--verbose_failures" {
                opts.verbose_failures = true;
                flag_map.insert(
                    "verbose_failures".to_string(),
                    "--verbose_failures=1".to_string(),
                );
                i += 1;
            } else if arg == "--noverbose_failures" {
                opts.verbose_failures = false;
                flag_map.insert(
                    "verbose_failures".to_string(),
                    "--verbose_failures=0".to_string(),
                );
                i += 1;
            } else if arg == "--expunge" {
                opts.expunge = true;
                flag_map.insert("expunge".to_string(), "--expunge=1".to_string());
                i += 1;
            } else if arg == "--expunge_async" {
                opts.expunge_async = true;
                flag_map.insert("expunge_async".to_string(), "--expunge_async=1".to_string());
                i += 1;
            } else if arg == "--show_make_env" {
                opts.show_make_env = true;
                flag_map.insert(
                    "show_make_env".to_string(),
                    "--show_make_env=1".to_string(),
                );
                i += 1;
            } else if let Some(val) = arg.strip_prefix("--output=") {
                opts.query_output_format = val.to_string();
                flag_map.insert("output".to_string(), format!("--output={val}"));
                i += 1;
            } else if arg == "--output" && i + 1 < args.len() {
                let val = &args[i + 1];
                opts.query_output_format = val.clone();
                flag_map.insert("output".to_string(), format!("--output={val}"));
                i += 2;
            } else if let Some(val) = arg.strip_prefix("--disk_cache=") {
                if !val.is_empty() {
                    opts.disk_cache = Some(PathFragment::create(val));
                }
                flag_map.insert("disk_cache".to_string(), format!("--disk_cache={val}"));
                i += 1;
            } else if let Some(val) = arg.strip_prefix("--profile=") {
                if !val.is_empty() {
                    opts.profile = Some(PathFragment::create(val));
                }
                flag_map.insert("profile".to_string(), format!("--profile={val}"));
                i += 1;
            } else if let Some(val) = arg.strip_prefix("--build_event_json_file=") {
                if !val.is_empty() {
                    opts.build_event_json_file = Some(PathFragment::create(val));
                }
                flag_map.insert(
                    "build_event_json_file".to_string(),
                    format!("--build_event_json_file={val}"),
                );
                i += 1;
            } else if let Some(val) = arg.strip_prefix("--test_output=") {
                opts.test_output = val.to_string();
                flag_map.insert("test_output".to_string(), format!("--test_output={val}"));
                i += 1;
            } else if let Some(val) = arg.strip_prefix("--test_filter=") {
                opts.test_filter = Some(val.to_string());
                flag_map.insert("test_filter".to_string(), format!("--test_filter={val}"));
                i += 1;
            } else if let Some(val) = arg.strip_prefix("--spawn_strategy=") {
                opts.spawn_strategy = val.to_string();
                flag_map.insert(
                    "spawn_strategy".to_string(),
                    format!("--spawn_strategy={val}"),
                );
                i += 1;
            } else if let Some(val) = arg.strip_prefix("--define=") {
                if let Some((dk, dv)) = val.split_once('=') {
                    opts.user_defines.insert(dk.to_string(), dv.to_string());
                }
                flag_map.insert(format!("define:{val}"), format!("--define={val}"));
                i += 1;
            } else if arg == "--define" && i + 1 < args.len() {
                let val = &args[i + 1];
                if let Some((dk, dv)) = val.split_once('=') {
                    opts.user_defines.insert(dk.to_string(), dv.to_string());
                }
                flag_map.insert(format!("define:{val}"), format!("--define={val}"));
                i += 2;
            } else if let Some(val) = arg.strip_prefix("--jobs=").or_else(|| arg.strip_prefix("-j=")) {
                let jobs = if val == "auto" {
                    8
                } else {
                    val.parse::<usize>().map_err(|_| {
                        OptionsParsingError(format!("Invalid --jobs value: '{val}'"))
                    })?
                };
                opts.jobs = jobs;
                flag_map.insert("jobs".to_string(), format!("--jobs={jobs}"));
                i += 1;
            } else if (arg == "-j" || arg == "--jobs") && i + 1 < args.len() {
                let val = &args[i + 1];
                let jobs = val.parse::<usize>().unwrap_or(8);
                opts.jobs = jobs;
                flag_map.insert("jobs".to_string(), format!("--jobs={jobs}"));
                i += 2;
            } else if let Some((k, v)) = arg.strip_prefix("--").and_then(|s| s.split_once('=')) {
                flag_map.insert(k.to_string(), format!("--{k}={v}"));
                i += 1;
            } else if let Some(no_key) = arg.strip_prefix("--no") {
                flag_map.insert(no_key.to_string(), format!("--{no_key}=0"));
                i += 1;
            } else if let Some(key) = arg.strip_prefix("--") {
                flag_map.insert(key.to_string(), format!("--{key}=1"));
                i += 1;
            } else {
                i += 1;
            }
        }

        opts.canonical_flags = flag_map.into_values().collect();
        Ok(opts)
    }

    fn validate_compilation_mode(val: &str) -> Result<(), OptionsParsingError> {
        match val {
            "fastbuild" | "dbg" | "opt" => Ok(()),
            _ => Err(OptionsParsingError(format!(
                "Not a valid compilation mode: '{val}' (should be fastbuild, dbg or opt)"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bazel_vfs::InMemoryFileSystem;

    #[test]
    fn test_split_command_line() {
        let args = vec![
            "--output_base=/tmp/ob".to_string(),
            "--batch".to_string(),
            "build".to_string(),
            "-c".to_string(),
            "opt".to_string(),
            "//src:bazel".to_string(),
        ];
        let (startup, cmd, cmd_args) = split_command_line(&args).unwrap();
        assert_eq!(
            startup.output_base,
            Some(PathFragment::create("/tmp/ob"))
        );
        assert!(startup.batch);
        assert_eq!(cmd, "build");
        assert_eq!(cmd_args, vec!["-c", "opt", "//src:bazel"]);
    }

    #[test]
    fn test_bazelrc_config_and_imports() {
        let fs = InMemoryFileSystem::default();
        let ws = PathFragment::create("/workspace");
        fs.create_directory_and_parents(&ws).unwrap();
        fs.write_file(
            &ws.get_relative_str("extra.bazelrc"),
            b"build:remote --jobs=64\n",
        )
        .unwrap();

        let main_rc = "\
import %workspace%/extra.bazelrc
common --verbose_failures
build -c opt
";
        let mut rc = BazelrcConfig::default();
        let mut visited = HashSet::new();
        rc.parse_str(main_rc, "/workspace/.bazelrc", &ws, &fs, &mut visited)
            .unwrap();

        let expanded = rc
            .expand_command_args(
                "build",
                &["--config=remote".to_string(), "//foo:bar".to_string()],
            )
            .unwrap();
        assert_eq!(
            expanded,
            vec!["--verbose_failures", "-c", "opt", "--jobs=64", "//foo:bar"]
        );

        let parsed = ParsedCommandOptions::parse(&expanded).unwrap();
        assert_eq!(parsed.compilation_mode, "opt");
        assert_eq!(parsed.jobs, 64);
        assert!(parsed.verbose_failures);
        assert_eq!(parsed.targets, vec!["//foo:bar"]);
    }
}

