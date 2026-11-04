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

//! Bazel Analysis Phase (`BuildConfigurationValue`, `Transition`, `Artifact`,
//! `ActionAnalysisMetadata`, `ConfiguredTarget`, and Core Providers).
//! Mirrors `src/main/java/com/google/devtools/build/lib/analysis` and `actions`.

use bazel_cmdline::Label;
use bazel_util::Fingerprint;
use bazel_vfs::PathFragment;
use std::collections::BTreeMap;
use std::fmt;

/// Key identifying a `BuildConfigurationValue` in Skyframe.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BuildConfigurationKey {
    pub cpu: String,
    pub compilation_mode: String,
    pub is_exec: bool,
    pub user_defines: BTreeMap<String, String>,
}

impl BuildConfigurationKey {
    pub fn target(cpu: impl Into<String>, compilation_mode: impl Into<String>) -> Self {
        Self {
            cpu: cpu.into(),
            compilation_mode: compilation_mode.into(),
            is_exec: false,
            user_defines: BTreeMap::new(),
        }
    }

    pub fn to_exec(&self) -> Self {
        Self {
            cpu: self.cpu.clone(),
            compilation_mode: "opt".to_string(),
            is_exec: true,
            user_defines: self.user_defines.clone(),
        }
    }

    pub fn mnemonic(&self) -> String {
        if self.is_exec {
            format!("{}-opt-exec", self.cpu)
        } else {
            format!("{}-{}", self.cpu, self.compilation_mode)
        }
    }

    pub fn short_hash(&self) -> String {
        let mut fp = Fingerprint::new();
        fp.add_str(&self.cpu)
            .add_str(&self.compilation_mode)
            .add_bool(self.is_exec);
        for (k, v) in &self.user_defines {
            fp.add_str(k).add_str(v);
        }
        let full = fp.hex_digest_and_reset();
        full[..7].to_string()
    }
}

/// Configuration transition kinds (`NoTransition`, `ExecTransition`, `TargetTransition`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigurationTransition {
    None,
    Exec,
    Target,
}

impl ConfigurationTransition {
    pub fn apply(&self, current: &BuildConfigurationKey) -> BuildConfigurationKey {
        match self {
            Self::None | Self::Target => current.clone(),
            Self::Exec => current.to_exec(),
        }
    }
}

/// Represents a source or generated file in the action graph, matching `Artifact.java`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Artifact {
    pub is_source: bool,
    pub is_tree_artifact: bool,
    /// Path relative to `$EXECROOT/_main` (e.g. `src/main.rs` or `bazel-out/darwin_arm64-fastbuild/bin/src/app`).
    pub exec_path: PathFragment,
    /// Path relative to the owning root (`src/main.rs` or `src/app`).
    pub root_relative_path: PathFragment,
    pub owner_label: Label,
}

impl Artifact {
    pub fn source(root_relative_path: PathFragment, owner_label: Label) -> Self {
        Self {
            is_source: true,
            is_tree_artifact: false,
            exec_path: root_relative_path.clone(),
            root_relative_path,
            owner_label,
        }
    }

    pub fn derived(
        config_mnemonic: &str,
        root_relative_path: PathFragment,
        owner_label: Label,
    ) -> Self {
        let exec_path = PathFragment::create(&format!(
            "bazel-out/{config_mnemonic}/bin/{root_relative_path}"
        ));
        Self {
            is_source: false,
            is_tree_artifact: false,
            exec_path,
            root_relative_path,
            owner_label,
        }
    }
}

impl fmt::Display for Artifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.exec_path)
    }
}

/// Action specification generated during the Analysis phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionAnalysisMetadata {
    SpawnAction {
        owner: Label,
        mnemonic: String,
        progress_message: String,
        argv: Vec<String>,
        env: BTreeMap<String, String>,
        inputs: Vec<Artifact>,
        outputs: Vec<Artifact>,
        execution_info: BTreeMap<String, String>,
    },
    FileWriteAction {
        owner: Label,
        output: Artifact,
        content: Vec<u8>,
        is_executable: bool,
    },
    SymlinkAction {
        owner: Label,
        input_path: PathFragment,
        output: Artifact,
        is_executable: bool,
    },
    TemplateExpansionAction {
        owner: Label,
        template: Artifact,
        output: Artifact,
        substitutions: BTreeMap<String, String>,
        is_executable: bool,
    },
}

impl ActionAnalysisMetadata {
    pub fn mnemonic(&self) -> &str {
        match self {
            Self::SpawnAction { mnemonic, .. } => mnemonic.as_str(),
            Self::FileWriteAction { .. } => "FileWrite",
            Self::SymlinkAction { .. } => "Symlink",
            Self::TemplateExpansionAction { .. } => "TemplateExpand",
        }
    }

    pub fn owner(&self) -> &Label {
        match self {
            Self::SpawnAction { owner, .. }
            | Self::FileWriteAction { owner, .. }
            | Self::SymlinkAction { owner, .. }
            | Self::TemplateExpansionAction { owner, .. } => owner,
        }
    }

    pub fn inputs(&self) -> Vec<Artifact> {
        match self {
            Self::SpawnAction { inputs, .. } => inputs.clone(),
            Self::TemplateExpansionAction { template, .. } => vec![template.clone()],
            Self::FileWriteAction { .. } | Self::SymlinkAction { .. } => Vec::new(),
        }
    }

    pub fn outputs(&self) -> Vec<Artifact> {
        match self {
            Self::SpawnAction { outputs, .. } => outputs.clone(),
            Self::FileWriteAction { output, .. }
            | Self::SymlinkAction { output, .. }
            | Self::TemplateExpansionAction { output, .. } => vec![output.clone()],
        }
    }

    /// Computes a deterministic SHA-256 action key over the action's mnemonic, command line,
    /// environment, and output paths.
    pub fn action_key(&self) -> String {
        let mut fp = Fingerprint::new();
        fp.add_str(self.mnemonic());
        match self {
            Self::SpawnAction {
                argv,
                env,
                inputs,
                outputs,
                ..
            } => {
                fp.add_strings(argv);
                for (k, v) in env {
                    fp.add_str(k).add_str(v);
                }
                for inp in inputs {
                    fp.add_str(inp.exec_path.path_string());
                }
                for out in outputs {
                    fp.add_str(out.exec_path.path_string());
                }
            }
            Self::FileWriteAction {
                output,
                content,
                is_executable,
                ..
            } => {
                fp.add_str(output.exec_path.path_string())
                    .add_bytes(content)
                    .add_bool(*is_executable);
            }
            Self::SymlinkAction {
                input_path,
                output,
                is_executable,
                ..
            } => {
                fp.add_str(input_path.path_string())
                    .add_str(output.exec_path.path_string())
                    .add_bool(*is_executable);
            }
            Self::TemplateExpansionAction {
                template,
                output,
                substitutions,
                is_executable,
                ..
            } => {
                fp.add_str(template.exec_path.path_string())
                    .add_str(output.exec_path.path_string())
                    .add_bool(*is_executable);
                for (k, v) in substitutions {
                    fp.add_str(k).add_str(v);
                }
            }
        }
        fp.hex_digest_and_reset()
    }

    pub fn argv(&self) -> Vec<String> {
        match self {
            Self::SpawnAction { argv, .. } => argv.clone(),
            _ => Vec::new(),
        }
    }

    pub fn pretty_print(&self) -> String {
        match self {
            Self::SpawnAction {
                progress_message, ..
            } => progress_message.clone(),
            Self::FileWriteAction { output, .. } => {
                format!("Writing file {}", output.exec_path)
            }
            Self::SymlinkAction { output, .. } => {
                format!("Symlinking {}", output.exec_path)
            }
            Self::TemplateExpansionAction { output, .. } => {
                format!("Expanding template {}", output.exec_path)
            }
        }
    }

    pub fn format_aquery_text(&self, config: &BuildConfigurationKey) -> String {
        let mut out = String::new();
        out.push_str(&format!("action '{}'\n", self.pretty_print()));
        out.push_str(&format!("  Mnemonic: {}\n", self.mnemonic()));
        out.push_str(&format!("  Target: {}\n", self.owner().canonical_form()));
        out.push_str(&format!(
            "  Configuration: {} ({})\n",
            config.mnemonic(),
            config.short_hash()
        ));
        out.push_str(&format!("  ActionKey: {}\n", self.action_key()));
        let inputs = self.inputs();
        if !inputs.is_empty() {
            let list: Vec<String> = inputs.iter().map(|a| a.exec_path.to_string()).collect();
            out.push_str(&format!("  Inputs: [{}]\n", list.join(", ")));
        }
        let outputs = self.outputs();
        if !outputs.is_empty() {
            let list: Vec<String> = outputs.iter().map(|a| a.exec_path.to_string()).collect();
            out.push_str(&format!("  Outputs: [{}]\n", list.join(", ")));
        }
        let argv = self.argv();
        if !argv.is_empty() {
            out.push_str(&format!("  Command Line: ({})\n", argv.join(" \\\n    ")));
        }
        out.push('\n');
        out
    }
}

/// Built-in `DefaultInfo` provider matching `DefaultInfo.java`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DefaultInfo {
    pub files: Vec<Artifact>,
    pub data_runfiles: Vec<Artifact>,
    pub default_runfiles: Vec<Artifact>,
    pub executable: Option<Artifact>,
}

/// Built-in `OutputGroupInfo` provider.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OutputGroupInfo {
    pub groups: BTreeMap<String, Vec<Artifact>>,
}

/// A configured target produced by evaluating a `Target` under a `BuildConfigurationKey`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfiguredTarget {
    pub label: Label,
    pub configuration: BuildConfigurationKey,
    pub rule_class: String,
    pub default_info: DefaultInfo,
    pub output_groups: OutputGroupInfo,
    pub actions: Vec<ActionAnalysisMetadata>,
    pub is_test: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_and_action_key_stability() {
        let cfg = BuildConfigurationKey::target("darwin_arm64", "opt");
        assert_eq!(cfg.mnemonic(), "darwin_arm64-opt");
        let exec_cfg = cfg.to_exec();
        assert_eq!(exec_cfg.mnemonic(), "darwin_arm64-opt-exec");

        let lbl = Label::parse_canonical("//src:app").unwrap();
        let out_art = Artifact::derived(
            &cfg.mnemonic(),
            PathFragment::create("src/app"),
            lbl.clone(),
        );
        let act1 = ActionAnalysisMetadata::FileWriteAction {
            owner: lbl.clone(),
            output: out_art.clone(),
            content: b"#!/bin/sh\necho hi\n".to_vec(),
            is_executable: true,
        };
        let act2 = act1.clone();
        assert_eq!(act1.action_key(), act2.action_key());
    }
}
