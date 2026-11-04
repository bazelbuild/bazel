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

//! Bazel Package model (`Package`, `Target`, `Rule`, `InputFile`, `OutputFile`) and Starlark
//! `BUILD` / `.bzl` evaluator.
//! Mirrors `src/main/java/com/google/devtools/build/lib/packages`.

use bazel_cmdline::{Label, PackageIdentifier};
use bazel_starlark::{EvalContext, LoadStatement, Value};
use bazel_vfs::{FileSystem, PathFragment};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleInstance {
    pub label: Label,
    pub rule_class: String,
    pub attrs: BTreeMap<String, Value>,
    pub srcs: Vec<String>,
    pub deps: Vec<String>,
    pub outs: Vec<String>,
    pub tools: Vec<String>,
    pub data: Vec<String>,
    pub visibility: Vec<String>,
    pub cmd: Option<String>,
    pub executable: bool,
    pub testonly: bool,
}

impl RuleInstance {
    pub fn is_test_rule(&self) -> bool {
        self.rule_class.ends_with("_test") || self.rule_class == "test_suite"
    }

    pub fn is_binary_rule(&self) -> bool {
        self.rule_class.ends_with("_binary") || self.executable
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    Rule(RuleInstance),
    InputFile {
        label: Label,
    },
    OutputFile {
        label: Label,
        generating_rule: Label,
    },
    PackageGroup {
        label: Label,
        packages: Vec<String>,
    },
}

impl Target {
    pub fn label(&self) -> &Label {
        match self {
            Self::Rule(r) => &r.label,
            Self::InputFile { label } => label,
            Self::OutputFile { label, .. } => label,
            Self::PackageGroup { label, .. } => label,
        }
    }

    pub fn target_kind(&self) -> String {
        match self {
            Self::Rule(r) => format!("{} rule", r.rule_class),
            Self::InputFile { .. } => "source file".to_string(),
            Self::OutputFile { .. } => "generated file".to_string(),
            Self::PackageGroup { .. } => "package group".to_string(),
        }
    }
}

/// An evaluated Bazel `BUILD` / `BUILD.bazel` package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Package {
    pub id: PackageIdentifier,
    pub build_file: PathFragment,
    pub targets: BTreeMap<String, Target>,
    pub loads: Vec<LoadStatement>,
    pub default_visibility: Vec<String>,
}

impl Package {
    /// Evaluates a `BUILD` file's Starlark source (and any local `.bzl` files it loads) into a `Package`.
    pub fn evaluate_build_file(
        id: PackageIdentifier,
        workspace_root: &PathFragment,
        build_file: PathFragment,
        source: &str,
        fs: Arc<dyn FileSystem>,
    ) -> Result<Self, String> {
        let pkg_dir = workspace_root.get_relative(id.package_fragment());
        let mut ctx = EvalContext::new(pkg_dir.clone(), Some(Arc::clone(&fs)));

        // Pre-evaluate loaded local `.bzl` files when present in the workspace
        let mut temp_ctx = EvalContext::new(pkg_dir.clone(), Some(Arc::clone(&fs)));
        let _ = temp_ctx.eval_str(source);
        for load in &temp_ctx.loads {
            if let Some(bzl_path) = resolve_local_bzl_path(workspace_root, id.package_fragment(), &load.module_label) {
                if let Ok(bzl_bytes) = fs.read_file(&bzl_path) {
                    let bzl_src = String::from_utf8_lossy(&bzl_bytes);
                    let mut bzl_ctx = EvalContext::new(
                        bzl_path.parent_directory().unwrap_or_else(PathFragment::empty),
                        Some(Arc::clone(&fs)),
                    );
                    if bzl_ctx.eval_str(&bzl_src).is_ok() {
                        for (local_sym, exported_sym) in &load.symbols {
                            if let Some(val) = bzl_ctx.globals.get(exported_sym) {
                                ctx.globals.insert(local_sym.clone(), val.clone());
                            }
                        }
                    }
                }
            }
        }

        ctx.eval_str(source).map_err(|e| e.to_string())?;

        let mut targets = BTreeMap::new();
        for exported in &ctx.exports_files {
            if let Ok(lbl) = Label::create(id.clone(), exported) {
                targets.insert(exported.clone(), Target::InputFile { label: lbl });
            }
        }

        for call in ctx.rule_calls {
            let Some(Value::String(name)) = call.kwargs.get("name").cloned() else {
                continue;
            };
            let Ok(label) = Label::create(id.clone(), &name) else {
                continue;
            };

            if call.rule_class == "package_group" {
                let packages = call
                    .kwargs
                    .get("packages")
                    .map(|v| v.to_string_list())
                    .unwrap_or_default();
                targets.insert(
                    name.clone(),
                    Target::PackageGroup {
                        label,
                        packages,
                    },
                );
                continue;
            }

            let srcs = extract_attr_strings(&call.kwargs, &["srcs", "src"]);
            let deps = extract_attr_strings(&call.kwargs, &["deps", "actual", "crate"]);
            let outs = extract_attr_strings(&call.kwargs, &["outs", "out"]);
            let tools = extract_attr_strings(&call.kwargs, &["tools", "tool"]);
            let data = extract_attr_strings(&call.kwargs, &["data"]);
            let visibility = extract_attr_strings(&call.kwargs, &["visibility"]);
            let cmd = call
                .kwargs
                .get("cmd")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let executable = call
                .kwargs
                .get("executable")
                .map(|v| v.is_truthy())
                .unwrap_or_else(|| call.rule_class.ends_with("_binary"));
            let testonly = call
                .kwargs
                .get("testonly")
                .map(|v| v.is_truthy())
                .unwrap_or(false);

            // Register source files and generated output files in the package target map
            for src in &srcs {
                let clean = src.strip_prefix(':').unwrap_or(src);
                if !clean.starts_with("//") && !clean.starts_with('@') && !targets.contains_key(clean) {
                    if let Ok(src_lbl) = Label::create(id.clone(), clean) {
                        targets.insert(
                            clean.to_string(),
                            Target::InputFile { label: src_lbl },
                        );
                    }
                }
            }
            for out_file in &outs {
                let clean = out_file.strip_prefix(':').unwrap_or(out_file);
                if let Ok(out_lbl) = Label::create(id.clone(), clean) {
                    targets.insert(
                        clean.to_string(),
                        Target::OutputFile {
                            label: out_lbl,
                            generating_rule: label.clone(),
                        },
                    );
                }
            }

            let rule = RuleInstance {
                label,
                rule_class: call.rule_class,
                attrs: call.kwargs,
                srcs,
                deps,
                outs,
                tools,
                data,
                visibility,
                cmd,
                executable,
                testonly,
            };
            targets.insert(name.clone(), Target::Rule(rule));
        }

        Ok(Self {
            id,
            build_file,
            targets,
            loads: ctx.loads,
            default_visibility: Vec::new(),
        })
    }

    pub fn rules(&self) -> impl Iterator<Item = &RuleInstance> {
        self.targets.values().filter_map(|t| match t {
            Target::Rule(r) => Some(r),
            _ => None,
        })
    }
}

fn extract_attr_strings(kwargs: &BTreeMap<String, Value>, keys: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for &k in keys {
        if let Some(val) = kwargs.get(k) {
            match val {
                Value::String(s) => out.push(s.clone()),
                Value::Label(l) => out.push(l.canonical_form()),
                _ => out.extend(val.to_string_list()),
            }
        }
    }
    out
}

fn resolve_local_bzl_path(
    workspace_root: &PathFragment,
    pkg_fragment: &PathFragment,
    module_label: &str,
) -> Option<PathFragment> {
    if module_label.starts_with('@') {
        return None;
    }
    if let Some(after_slashes) = module_label.strip_prefix("//") {
        let (pkg_part, file_part) = after_slashes.split_once(':')?;
        Some(
            workspace_root
                .get_relative_str(pkg_part)
                .get_relative_str(file_part),
        )
    } else if let Some(rel_file) = module_label.strip_prefix(':') {
        Some(
            workspace_root
                .get_relative(pkg_fragment)
                .get_relative_str(rel_file),
        )
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bazel_vfs::InMemoryFileSystem;

    #[test]
    fn test_package_bzl_load_and_macro_expansion() {
        let fs = Arc::new(InMemoryFileSystem::default());
        let ws = PathFragment::create("/workspace");
        let pkg_dir = ws.get_relative_str("src");
        fs.create_directory_and_parents(&pkg_dir).unwrap();

        fs.write_file(
            &pkg_dir.get_relative_str("macros.bzl"),
            br#"
def declare_pair(base):
    rust_library(
        name = base + "_lib",
        srcs = [base + ".rs"],
    )
    rust_test(
        name = base + "_test",
        crate = ":" + base + "_lib",
    )
"#,
        )
        .unwrap();

        let build_src = r#"
load(":macros.bzl", "declare_pair")
declare_pair("engine")
"#;
        let pkg_id = PackageIdentifier::create_in_main_repo("src").unwrap();
        let pkg = Package::evaluate_build_file(
            pkg_id,
            &ws,
            pkg_dir.get_relative_str("BUILD"),
            build_src,
            fs,
        )
        .unwrap();

        let rule_names: Vec<&str> = pkg.rules().map(|r| r.label.name()).collect();
        assert_eq!(rule_names, vec!["engine_lib", "engine_test"]);
        assert!(pkg.targets.contains_key("engine.rs"));
    }
}
