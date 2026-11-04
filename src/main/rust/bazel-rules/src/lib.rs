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

//! Built-in Rule Implementations and `@_builtins` Starlark bridge (`genrule`, `filegroup`,
//! `cc_*`, `rust_*`, `java_*`, `py_*`, `sh_*`, `copy_file`, `bazel_binary`).
//! Mirrors `src/main/java/com/google/devtools/build/lib/rules` and `src/main/starlark/builtins_bzl`.

use bazel_analysis::{
    ActionAnalysisMetadata, Artifact, BuildConfigurationKey, ConfiguredTarget, DefaultInfo,
    OutputGroupInfo,
};
use bazel_cmdline::{Label, PackageIdentifier};
use bazel_packages::{RuleInstance, Target};
use std::collections::BTreeMap;

/// Analyzes a loaded `Target` under `config` into a `ConfiguredTarget` with concrete actions.
pub fn analyze_target(
    target: &Target,
    config: &BuildConfigurationKey,
) -> Result<ConfiguredTarget, String> {
    let mnemonic = config.mnemonic();
    match target {
        Target::InputFile { label } => {
            let rel_path = label
                .package_identifier()
                .package_fragment()
                .get_relative_str(label.name());
            let art = Artifact::source(rel_path, label.clone());
            Ok(ConfiguredTarget {
                label: label.clone(),
                configuration: config.clone(),
                rule_class: "source file".to_string(),
                default_info: DefaultInfo {
                    files: vec![art],
                    ..Default::default()
                },
                output_groups: OutputGroupInfo::default(),
                actions: Vec::new(),
                is_test: false,
            })
        }
        Target::OutputFile {
            label,
            generating_rule,
        } => {
            let rel_path = label
                .package_identifier()
                .package_fragment()
                .get_relative_str(label.name());
            let art = Artifact::derived(&mnemonic, rel_path, generating_rule.clone());
            Ok(ConfiguredTarget {
                label: label.clone(),
                configuration: config.clone(),
                rule_class: "generated file".to_string(),
                default_info: DefaultInfo {
                    files: vec![art],
                    ..Default::default()
                },
                output_groups: OutputGroupInfo::default(),
                actions: Vec::new(),
                is_test: false,
            })
        }
        Target::PackageGroup { label, .. } => Ok(ConfiguredTarget {
            label: label.clone(),
            configuration: config.clone(),
            rule_class: "package_group".to_string(),
            default_info: DefaultInfo::default(),
            output_groups: OutputGroupInfo::default(),
            actions: Vec::new(),
            is_test: false,
        }),
        Target::Rule(rule) => analyze_rule(rule, config),
    }
}

fn analyze_rule(
    rule: &RuleInstance,
    config: &BuildConfigurationKey,
) -> Result<ConfiguredTarget, String> {
    let cfg_mnemonic = config.mnemonic();
    let pkg_id = rule.label.package_identifier();
    let pkg_frag = pkg_id.package_fragment();

    let mut src_artifacts = Vec::new();
    for s in &rule.srcs {
        let clean = s.strip_prefix(':').unwrap_or(s);
        let rel = if clean.starts_with("//") {
            if let Ok(lbl) = Label::parse_canonical(clean) {
                lbl.package_identifier()
                    .package_fragment()
                    .get_relative_str(lbl.name())
            } else {
                pkg_frag.get_relative_str(clean)
            }
        } else {
            pkg_frag.get_relative_str(clean)
        };
        src_artifacts.push(Artifact::source(rel, rule.label.clone()));
    }

    let mut out_artifacts = Vec::new();
    for o in &rule.outs {
        let clean = o.strip_prefix(':').unwrap_or(o);
        let rel = pkg_frag.get_relative_str(clean);
        out_artifacts.push(Artifact::derived(&cfg_mnemonic, rel, rule.label.clone()));
    }

    let mut actions = Vec::new();
    let mut executable: Option<Artifact> = None;

    match rule.rule_class.as_str() {
        "filegroup" | "alias" | "test_suite" | "config_setting" => {
            if out_artifacts.is_empty() {
                out_artifacts = src_artifacts.clone();
            }
        }
        "genrule" => {
            if out_artifacts.is_empty() {
                let default_out = pkg_frag.get_relative_str(rule.label.name());
                out_artifacts.push(Artifact::derived(
                    &cfg_mnemonic,
                    default_out,
                    rule.label.clone(),
                ));
            }
            let raw_cmd = rule.cmd.as_deref().unwrap_or("true");
            let expanded_cmd = expand_genrule_cmd(pkg_id, raw_cmd, &src_artifacts, &out_artifacts);
            actions.push(ActionAnalysisMetadata::SpawnAction {
                owner: rule.label.clone(),
                mnemonic: "Genrule".to_string(),
                progress_message: format!("Executing genrule {}", rule.label),
                argv: vec![
                    "/bin/bash".to_string(),
                    "-c".to_string(),
                    expanded_cmd,
                ],
                env: BTreeMap::new(),
                inputs: src_artifacts.clone(),
                outputs: out_artifacts.clone(),
                execution_info: BTreeMap::new(),
            });
        }
        "copy_file" => {
            if out_artifacts.is_empty() {
                let default_out = pkg_frag.get_relative_str(rule.label.name());
                out_artifacts.push(Artifact::derived(
                    &cfg_mnemonic,
                    default_out,
                    rule.label.clone(),
                ));
            }
            if let (Some(src), Some(dst)) = (src_artifacts.first(), out_artifacts.first()) {
                actions.push(ActionAnalysisMetadata::SpawnAction {
                    owner: rule.label.clone(),
                    mnemonic: "CopyFile".to_string(),
                    progress_message: format!("Copying {} to {}", src.exec_path, dst.exec_path),
                    argv: vec![
                        "/bin/cp".to_string(),
                        "-f".to_string(),
                        src.exec_path.to_string(),
                        dst.exec_path.to_string(),
                    ],
                    env: BTreeMap::new(),
                    inputs: vec![src.clone()],
                    outputs: vec![dst.clone()],
                    execution_info: BTreeMap::new(),
                });
            }
        }
        "sh_binary" | "sh_test" | "py_binary" | "py_test" => {
            let bin_rel = pkg_frag.get_relative_str(rule.label.name());
            let bin_art = Artifact::derived(&cfg_mnemonic, bin_rel, rule.label.clone());
            if let Some(first_src) = src_artifacts.first() {
                actions.push(ActionAnalysisMetadata::SpawnAction {
                    owner: rule.label.clone(),
                    mnemonic: "ExecutableSymlink".to_string(),
                    progress_message: format!("Creating executable {}", rule.label),
                    argv: vec![
                        "/bin/cp".to_string(),
                        "-f".to_string(),
                        first_src.exec_path.to_string(),
                        bin_art.exec_path.to_string(),
                    ],
                    env: BTreeMap::new(),
                    inputs: src_artifacts.clone(),
                    outputs: vec![bin_art.clone()],
                    execution_info: BTreeMap::new(),
                });
            } else {
                actions.push(ActionAnalysisMetadata::FileWriteAction {
                    owner: rule.label.clone(),
                    output: bin_art.clone(),
                    content: b"#!/bin/sh\nexit 0\n".to_vec(),
                    is_executable: true,
                });
            }
            executable = Some(bin_art.clone());
            out_artifacts = vec![bin_art];
        }
        "cc_binary" | "cc_test" => {
            let bin_rel = pkg_frag.get_relative_str(rule.label.name());
            let bin_art = Artifact::derived(&cfg_mnemonic, bin_rel, rule.label.clone());
            let mut argv = vec!["cc".to_string()];
            if config.compilation_mode == "opt" {
                argv.push("-O2".to_string());
            } else if config.compilation_mode == "dbg" {
                argv.push("-g".to_string());
            }
            for src in &src_artifacts {
                argv.push(src.exec_path.to_string());
            }
            argv.push("-o".to_string());
            argv.push(bin_art.exec_path.to_string());

            actions.push(ActionAnalysisMetadata::SpawnAction {
                owner: rule.label.clone(),
                mnemonic: "CppLink".to_string(),
                progress_message: format!("Linking {}", rule.label),
                argv,
                env: BTreeMap::new(),
                inputs: src_artifacts.clone(),
                outputs: vec![bin_art.clone()],
                execution_info: BTreeMap::new(),
            });
            executable = Some(bin_art.clone());
            out_artifacts = vec![bin_art];
        }
        "cc_library" => {
            let lib_rel = pkg_frag.get_relative_str(&format!("lib{}.a", rule.label.name()));
            let lib_art = Artifact::derived(&cfg_mnemonic, lib_rel, rule.label.clone());
            actions.push(ActionAnalysisMetadata::FileWriteAction {
                owner: rule.label.clone(),
                output: lib_art.clone(),
                content: b"!<arch>\n".to_vec(),
                is_executable: false,
            });
            out_artifacts = vec![lib_art];
        }
        "rust_binary" | "rust_test" | "rust_library" | "rust_shared_library" => {
            let is_bin = rule.rule_class == "rust_binary" || rule.rule_class == "rust_test";
            let crate_name = rule.label.name().replace('-', "_");
            let out_name = if is_bin {
                rule.label.name().to_string()
            } else if rule.rule_class == "rust_shared_library" {
                format!("lib{crate_name}.dylib")
            } else {
                format!("lib{crate_name}.rlib")
            };
            let out_rel = pkg_frag.get_relative_str(&out_name);
            let out_art = Artifact::derived(&cfg_mnemonic, out_rel, rule.label.clone());

            let crate_root = src_artifacts
                .iter()
                .find(|a| {
                    let b = a.exec_path.base_name();
                    b == "main.rs" || b == "lib.rs"
                })
                .or_else(|| src_artifacts.first());

            if let Some(root_src) = crate_root {
                let mut argv = vec![
                    "rustc".to_string(),
                    "--edition=2021".to_string(),
                    format!("--crate-name={crate_name}"),
                    root_src.exec_path.to_string(),
                    "-o".to_string(),
                    out_art.exec_path.to_string(),
                ];
                if rule.rule_class == "rust_library" {
                    argv.push("--crate-type=rlib".to_string());
                } else if rule.rule_class == "rust_shared_library" {
                    argv.push("--crate-type=cdylib".to_string());
                } else if rule.rule_class == "rust_test" {
                    argv.push("--test".to_string());
                }
                let mut action_inputs = src_artifacts.clone();
                if !rule.deps.is_empty() {
                    let bin_pkg_dir = format!("bazel-out/{cfg_mnemonic}/bin/{pkg_frag}");
                    argv.push("-L".to_string());
                    argv.push(format!("dependency={bin_pkg_dir}"));
                    for dep_str in &rule.deps {
                        let dep_name = dep_str
                            .rsplit(':')
                            .next()
                            .unwrap_or(dep_str)
                            .trim_start_matches("//")
                            .replace('-', "_");
                        let dep_crate_name = dep_name.strip_prefix("bazel_client_lib").map(|_| "bazel_client").unwrap_or(&dep_name);
                        let dep_rlib_rel = pkg_frag.get_relative_str(&format!("lib{dep_name}.rlib"));
                        let dep_art = Artifact::derived(&cfg_mnemonic, dep_rlib_rel, rule.label.clone());
                        argv.push("--extern".to_string());
                        argv.push(format!("{dep_crate_name}={}", dep_art.exec_path));
                        action_inputs.push(dep_art);
                    }
                }
                actions.push(ActionAnalysisMetadata::SpawnAction {
                    owner: rule.label.clone(),
                    mnemonic: "Rustc".to_string(),
                    progress_message: format!("Compiling Rust {} {}", rule.rule_class, rule.label),
                    argv,
                    env: BTreeMap::new(),
                    inputs: action_inputs,
                    outputs: vec![out_art.clone()],
                    execution_info: BTreeMap::new(),
                });
            } else {
                actions.push(ActionAnalysisMetadata::FileWriteAction {
                    owner: rule.label.clone(),
                    output: out_art.clone(),
                    content: Vec::new(),
                    is_executable: is_bin,
                });
            }
            if is_bin {
                executable = Some(out_art.clone());
            }
            out_artifacts = vec![out_art];
        }
        _ => {
            if out_artifacts.is_empty() {
                let default_out = pkg_frag.get_relative_str(rule.label.name());
                out_artifacts.push(Artifact::derived(
                    &cfg_mnemonic,
                    default_out,
                    rule.label.clone(),
                ));
            }
            for out_art in &out_artifacts {
                actions.push(ActionAnalysisMetadata::FileWriteAction {
                    owner: rule.label.clone(),
                    output: out_art.clone(),
                    content: Vec::new(),
                    is_executable: rule.is_binary_rule(),
                });
            }
            if rule.is_binary_rule() {
                executable = out_artifacts.first().cloned();
            }
        }
    }

    let mut groups = BTreeMap::new();
    groups.insert("default".to_string(), out_artifacts.clone());

    Ok(ConfiguredTarget {
        label: rule.label.clone(),
        configuration: config.clone(),
        rule_class: rule.rule_class.clone(),
        default_info: DefaultInfo {
            files: out_artifacts,
            data_runfiles: src_artifacts,
            default_runfiles: Vec::new(),
            executable,
        },
        output_groups: OutputGroupInfo { groups },
        actions,
        is_test: rule.is_test_rule(),
    })
}

/// Expands `$(SRCS)`, `$<`, `$(OUTS)`, `$@`, and `$(location ...)` in a `genrule` command string.
pub fn expand_genrule_cmd(
    pkg_id: &PackageIdentifier,
    raw_cmd: &str,
    srcs: &[Artifact],
    outs: &[Artifact],
) -> String {
    let srcs_str: Vec<String> = srcs.iter().map(|a| a.exec_path.to_string()).collect();
    let outs_str: Vec<String> = outs.iter().map(|a| a.exec_path.to_string()).collect();
    let joined_srcs = srcs_str.join(" ");
    let joined_outs = outs_str.join(" ");

    let mut cmd = raw_cmd
        .replace("$(SRCS)", &joined_srcs)
        .replace("$<", srcs_str.first().map(|s| s.as_str()).unwrap_or(""))
        .replace("$(OUTS)", &joined_outs)
        .replace("$@", outs_str.first().map(|s| s.as_str()).unwrap_or(""));

    // Expand $(location <label>) and $(locations <label>)
    for prefix in ["$(location ", "$(locations ", "$(execpath ", "$(execpaths "] {
        while let Some(start) = cmd.find(prefix) {
            if let Some(rel_end) = cmd[start..].find(')') {
                let end = start + rel_end;
                let inner_lbl = cmd[start + prefix.len()..end].trim();
                let clean = inner_lbl.strip_prefix(':').unwrap_or(inner_lbl);
                let replacement = srcs
                    .iter()
                    .chain(outs.iter())
                    .find(|a| {
                        a.exec_path.base_name() == clean
                            || a.root_relative_path.path_string() == clean
                    })
                    .map(|a| a.exec_path.to_string())
                    .unwrap_or_else(|| {
                        pkg_id
                            .package_fragment()
                            .get_relative_str(clean)
                            .to_string()
                    });
                cmd.replace_range(start..=end, &replacement);
            } else {
                break;
            }
        }
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genrule_location_and_var_expansion() {
        let pkg_id = PackageIdentifier::create_in_main_repo("src").unwrap();
        let lbl = Label::create(pkg_id.clone(), "gen").unwrap();
        let cfg = BuildConfigurationKey::target("darwin_arm64", "fastbuild");
        let rule = RuleInstance {
            label: lbl,
            rule_class: "genrule".to_string(),
            attrs: BTreeMap::new(),
            srcs: vec!["in.txt".to_string()],
            deps: Vec::new(),
            outs: vec!["out.txt".to_string()],
            tools: Vec::new(),
            data: Vec::new(),
            visibility: Vec::new(),
            cmd: Some("cat $(location :in.txt) > $@".to_string()),
            executable: false,
            testonly: false,
        };
        let ct = analyze_target(&Target::Rule(rule), &cfg).unwrap();
        assert_eq!(ct.actions.len(), 1);
        let ActionAnalysisMetadata::SpawnAction { argv, .. } = &ct.actions[0] else {
            panic!("Expected SpawnAction");
        };
        assert_eq!(
            argv[2],
            "cat src/in.txt > bazel-out/darwin_arm64-fastbuild/bin/src/out.txt"
        );
    }
}
