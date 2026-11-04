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

//! Bzlmod External Dependency System (`MODULE.bazel` evaluation, Minimum Version Selection,
//! overrides, `RepositoryMapping`, and `MODULE.bazel.lock`).
//! Mirrors `src/main/java/com/google/devtools/build/lib/bazel/bzlmod`.

use bazel_cmdline::{RepositoryMapping, RepositoryName};
use bazel_starlark::{EvalContext, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::fmt;

/// Semantic version comparator supporting BCR versions (e.g. `1.8.2.bcr.1`, `10.0.0-prerelease`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Version(pub String);

impl Version {
    pub fn parse(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0 == other.0 {
            return Ordering::Equal;
        }
        let split_parts = |s: &str| -> Vec<String> {
            s.split(|c: char| c == '.' || c == '-')
                .map(|p| p.to_string())
                .collect()
        };
        let a = split_parts(&self.0);
        let b = split_parts(&other.0);
        for (pa, pb) in a.iter().zip(b.iter()) {
            let cmp = match (pa.parse::<u64>(), pb.parse::<u64>()) {
                (Ok(na), Ok(nb)) => na.cmp(&nb),
                _ => pa.cmp(pb),
            };
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        a.len().cmp(&b.len())
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModuleKey {
    pub name: String,
    pub version: Version,
}

impl ModuleKey {
    pub fn canonical_repo_name(&self, is_root: bool) -> RepositoryName {
        if is_root {
            RepositoryName::main()
        } else {
            RepositoryName::create_unvalidated(format!("{}+{}", self.name, self.version))
        }
    }
}

impl fmt::Display for ModuleKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.version.is_empty() {
            f.write_str(&self.name)
        } else {
            write!(f, "{}@{}", self.name, self.version)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BazelDepSpec {
    pub name: String,
    pub version: Version,
    pub repo_name: Option<String>,
    pub dev_dependency: bool,
    pub max_compatibility_level: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleOverride {
    SingleVersion {
        version: Version,
        patches: Vec<String>,
    },
    LocalPath {
        path: String,
    },
    Archive {
        urls: Vec<String>,
        integrity: String,
    },
}

/// Parsed representation of a `MODULE.bazel` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterimModule {
    pub name: String,
    pub version: Version,
    pub repo_name: String,
    pub compatibility_level: i64,
    pub deps: Vec<BazelDepSpec>,
    pub overrides: BTreeMap<String, ModuleOverride>,
    pub registered_toolchains: Vec<String>,
    pub registered_execution_platforms: Vec<String>,
}

impl InterimModule {
    /// Evaluates a `MODULE.bazel` file using `bazel-starlark`.
    pub fn parse(content: &str) -> Result<Self, String> {
        let mut ctx = EvalContext::default();
        ctx.eval_str(content).map_err(|e| e.to_string())?;

        let mut name = "_main".to_string();
        let mut version = Version::parse("0.0.0");
        let mut repo_name = String::new();
        let mut compatibility_level = 0i64;
        let mut deps = Vec::new();
        let mut overrides = BTreeMap::new();
        let registered_toolchains = Vec::new();
        let registered_execution_platforms = Vec::new();

        for call in ctx.rule_calls {
            match call.rule_class.as_str() {
                "module" => {
                    if let Some(Value::String(n)) = call.kwargs.get("name") {
                        name = n.clone();
                    }
                    if let Some(Value::String(v)) = call.kwargs.get("version") {
                        version = Version::parse(v.clone());
                    }
                    if let Some(Value::String(rn)) = call.kwargs.get("repo_name") {
                        repo_name = rn.clone();
                    }
                    if let Some(Value::Int(cl)) = call.kwargs.get("compatibility_level") {
                        compatibility_level = *cl;
                    }
                }
                "bazel_dep" => {
                    let Some(Value::String(dep_name)) = call.kwargs.get("name") else {
                        continue;
                    };
                    let dep_ver = call
                        .kwargs
                        .get("version")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let dep_repo_name = match call.kwargs.get("repo_name") {
                        Some(Value::String(rn)) => Some(rn.clone()),
                        Some(Value::None) => None,
                        _ => Some(dep_name.clone()),
                    };
                    let dev_dependency = call
                        .kwargs
                        .get("dev_dependency")
                        .map(|v| v.is_truthy())
                        .unwrap_or(false);
                    let max_compatibility_level = call
                        .kwargs
                        .get("max_compatibility_level")
                        .and_then(|v| match v {
                            Value::Int(i) => Some(*i),
                            _ => None,
                        })
                        .unwrap_or(-1);
                    deps.push(BazelDepSpec {
                        name: dep_name.clone(),
                        version: Version::parse(dep_ver),
                        repo_name: dep_repo_name,
                        dev_dependency,
                        max_compatibility_level,
                    });
                }
                "single_version_override" => {
                    if let Some(Value::String(mod_name)) = call.kwargs.get("module_name") {
                        let ver = call
                            .kwargs
                            .get("version")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let patches = call
                            .kwargs
                            .get("patches")
                            .map(|v| v.to_string_list())
                            .unwrap_or_default();
                        overrides.insert(
                            mod_name.clone(),
                            ModuleOverride::SingleVersion {
                                version: Version::parse(ver),
                                patches,
                            },
                        );
                    }
                }
                "local_path_override" => {
                    if let Some(Value::String(mod_name)) = call.kwargs.get("module_name") {
                        let path = call
                            .kwargs
                            .get("path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        overrides.insert(mod_name.clone(), ModuleOverride::LocalPath { path });
                    }
                }
                "archive_override" => {
                    if let Some(Value::String(mod_name)) = call.kwargs.get("module_name") {
                        let urls = call
                            .kwargs
                            .get("urls")
                            .map(|v| v.to_string_list())
                            .unwrap_or_default();
                        let integrity = call
                            .kwargs
                            .get("integrity")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        overrides.insert(
                            mod_name.clone(),
                            ModuleOverride::Archive { urls, integrity },
                        );
                    }
                }
                _ => {}
            }
        }

        if repo_name.is_empty() {
            repo_name = name.clone();
        }

        Ok(Self {
            name,
            version,
            repo_name,
            compatibility_level,
            deps,
            overrides,
            registered_toolchains,
            registered_execution_platforms,
        })
    }
}

/// Result of running Bzlmod Minimum Version Selection (MVS).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModuleGraph {
    pub root: InterimModule,
    pub selected_versions: BTreeMap<String, Version>,
    pub root_repo_mapping: RepositoryMapping,
}

impl ResolvedModuleGraph {
    /// Runs Minimum Version Selection (MVS) over the root module and transitive registry modules,
    /// applying `single_version_override`s.
    pub fn resolve(
        root: InterimModule,
        transitive_modules: &[InterimModule],
    ) -> Self {
        let mut selected: BTreeMap<String, Version> = BTreeMap::new();

        for dep in &root.deps {
            let entry = selected
                .entry(dep.name.clone())
                .or_insert_with(|| dep.version.clone());
            if dep.version > *entry {
                *entry = dep.version.clone();
            }
        }

        for m in transitive_modules {
            let entry = selected
                .entry(m.name.clone())
                .or_insert_with(|| m.version.clone());
            if m.version > *entry {
                *entry = m.version.clone();
            }
            for dep in &m.deps {
                let d_entry = selected
                    .entry(dep.name.clone())
                    .or_insert_with(|| dep.version.clone());
                if dep.version > *d_entry {
                    *d_entry = dep.version.clone();
                }
            }
        }

        // Apply single_version_override from root module
        for (mod_name, ovr) in &root.overrides {
            if let ModuleOverride::SingleVersion { version, .. } = ovr {
                if !version.is_empty() {
                    selected.insert(mod_name.clone(), version.clone());
                }
            }
        }

        let mut mapping_entries = HashMap::new();
        mapping_entries.insert(root.repo_name.clone(), RepositoryName::main());
        mapping_entries.insert(root.name.clone(), RepositoryName::main());
        for dep in &root.deps {
            if let Some(ref apparent) = dep.repo_name {
                let ver = selected
                    .get(&dep.name)
                    .cloned()
                    .unwrap_or_else(|| dep.version.clone());
                let canonical = if ver.is_empty() {
                    RepositoryName::create_unvalidated(format!("{}+", dep.name))
                } else {
                    RepositoryName::create_unvalidated(format!("{}+{}", dep.name, ver))
                };
                mapping_entries.insert(apparent.clone(), canonical);
            }
        }

        let root_repo_mapping = RepositoryMapping::new(mapping_entries, RepositoryName::main());
        Self {
            root,
            selected_versions: selected,
            root_repo_mapping,
        }
    }

    pub fn format_graph_text(&self) -> String {
        let mut out = format!("<root> ({}@{})\n", self.root.name, self.root.version);
        for (i, dep) in self.root.deps.iter().enumerate() {
            let branch = if i + 1 == self.root.deps.len() {
                "└──"
            } else {
                "├──"
            };
            let ver = self
                .selected_versions
                .get(&dep.name)
                .unwrap_or(&dep.version);
            if ver.is_empty() {
                out.push_str(&format!("{branch} {}\n", dep.name));
            } else {
                out.push_str(&format!("{branch} {}@{}\n", dep.name, ver));
            }
        }
        out
    }

    pub fn format_show_repo(&self, repo_arg: &str) -> String {
        let clean = repo_arg.trim_start_matches('@');
        let ver = self
            .selected_versions
            .get(clean)
            .map(|v| v.0.as_str())
            .unwrap_or("0.0.0");
        format!("## {clean}:\nhttp_archive(\n  name = \"{clean}+\",\n  version = \"{ver}\",\n)\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bzlmod_mvs_and_overrides() {
        let root_src = r#"
module(
    name = "bazel",
    version = "10.0.0-prerelease",
    repo_name = "io_bazel",
)
bazel_dep(name = "rules_rs", version = "0.1.3")
bazel_dep(name = "protobuf", version = "35.0", repo_name = "com_google_protobuf")
single_version_override(
    module_name = "protobuf",
    version = "36.2",
)
"#;
        let root_mod = InterimModule::parse(root_src).unwrap();
        assert_eq!(root_mod.name, "bazel");
        assert_eq!(root_mod.repo_name, "io_bazel");

        let transitive_mod = InterimModule::parse(
            r#"
module(name = "rules_rs", version = "0.1.3")
bazel_dep(name = "platforms", version = "1.1.0")
"#,
        )
        .unwrap();

        let graph = ResolvedModuleGraph::resolve(root_mod, &[transitive_mod]);
        assert_eq!(
            graph.selected_versions.get("protobuf"),
            Some(&Version::parse("36.2"))
        );
        assert_eq!(
            graph.selected_versions.get("platforms"),
            Some(&Version::parse("1.1.0"))
        );
        assert_eq!(
            graph.root_repo_mapping.get("com_google_protobuf").name(),
            "protobuf+36.2"
        );
        assert!(graph.root_repo_mapping.get("io_bazel").is_main());
    }
}
