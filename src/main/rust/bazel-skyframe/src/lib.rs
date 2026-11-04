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

//! Skyframe Incremental Evaluation Engine (`SkyKey`, `SkyValue`, `SkyFunction`, `InMemoryMemoizingEvaluator`).
//! Mirrors `src/main/java/com/google/devtools/build/skyframe` and `build/lib/skyframe`.

use bazel_analysis::{BuildConfigurationKey, ConfiguredTarget};
use bazel_cmdline::{Label, PackageIdentifier};
use bazel_packages::{Package, Target};
use bazel_rules::analyze_target;
use bazel_vfs::{FileSystem, PathFragment, RootedPath};
use std::any::Any;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SkyFunctionName(pub &'static str);

impl SkyFunctionName {
    pub const FILE_STATE: Self = Self("FILE_STATE");
    pub const FILE: Self = Self("FILE");
    pub const DIRECTORY_LISTING: Self = Self("DIRECTORY_LISTING");
    pub const PACKAGE_LOOKUP: Self = Self("PACKAGE_LOOKUP");
    pub const BZL_LOAD: Self = Self("BZL_LOAD");
    pub const PACKAGE: Self = Self("PACKAGE");
    pub const CONFIGURED_TARGET: Self = Self("CONFIGURED_TARGET");
    pub const ACTION_EXECUTION: Self = Self("ACTION_EXECUTION");
    pub const PRECOMPUTED: Self = Self("PRECOMPUTED");
}

impl fmt::Display for SkyFunctionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// Canonical `SkyKey` representation across core Bazel SkyFunctions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SkyKey {
    FileState(RootedPath),
    PackageLookup(PackageIdentifier),
    Package(PackageIdentifier),
    ConfiguredTarget { label: Label, config_hash: String },
    Precomputed(String),
    Custom {
        function_name: SkyFunctionName,
        argument: String,
    },
}

impl SkyKey {
    pub fn function_name(&self) -> SkyFunctionName {
        match self {
            Self::FileState(_) => SkyFunctionName::FILE_STATE,
            Self::PackageLookup(_) => SkyFunctionName::PACKAGE_LOOKUP,
            Self::Package(_) => SkyFunctionName::PACKAGE,
            Self::ConfiguredTarget { .. } => SkyFunctionName::CONFIGURED_TARGET,
            Self::Precomputed(_) => SkyFunctionName::PRECOMPUTED,
            Self::Custom { function_name, .. } => *function_name,
        }
    }
}

impl fmt::Display for SkyKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileState(rp) => write!(f, "FILE_STATE:[{}]", rp.as_path()),
            Self::PackageLookup(pkg) => write!(f, "PACKAGE_LOOKUP:[{pkg}]"),
            Self::Package(pkg) => write!(f, "PACKAGE:[{pkg}]"),
            Self::ConfiguredTarget { label, config_hash } => {
                write!(f, "CONFIGURED_TARGET:[{label}@{config_hash}]")
            }
            Self::Precomputed(name) => write!(f, "PRECOMPUTED:[{name}]"),
            Self::Custom {
                function_name,
                argument,
            } => write!(f, "{function_name}:[{argument}]"),
        }
    }
}

/// Trait implemented by all Skyframe values.
pub trait SkyValue: Any + fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn eq_value(&self, other: &dyn SkyValue) -> bool;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileStateValue {
    RegularFile { size: u64, digest: Vec<u8> },
    Directory,
    Symlink { target: PathFragment },
    NonExistent,
}

impl SkyValue for FileStateValue {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn eq_value(&self, other: &dyn SkyValue) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageLookupValue {
    pub package_exists: bool,
    pub build_file_path: Option<RootedPath>,
}

impl SkyValue for PackageLookupValue {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn eq_value(&self, other: &dyn SkyValue) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }
}

/// Represents a loaded target inside a `PackageValue`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleTarget {
    pub label: Label,
    pub rule_class: String,
    pub srcs: Vec<String>,
    pub deps: Vec<String>,
    pub cmd: Option<String>,
    pub outs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageValue {
    pub package_id: PackageIdentifier,
    pub build_file: RootedPath,
    pub targets: BTreeMap<String, RuleTarget>,
    pub package: Option<Package>,
}

impl SkyValue for PackageValue {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn eq_value(&self, other: &dyn SkyValue) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfiguredTargetValue {
    pub configured_target: ConfiguredTarget,
}

impl SkyValue for ConfiguredTargetValue {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn eq_value(&self, other: &dyn SkyValue) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringSkyValue(pub String);

impl SkyValue for StringSkyValue {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn eq_value(&self, other: &dyn SkyValue) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkyFunctionError {
    pub message: String,
    pub transient: bool,
}

impl SkyFunctionError {
    pub fn persistent(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            transient: false,
        }
    }

    pub fn transient(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            transient: true,
        }
    }
}

impl fmt::Display for SkyFunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

/// `SkyFunction.Environment` passed to `SkyFunction::compute`.
/// Mirrors Bazel's `env.getValue()` / `env.valuesMissing()` contract.
pub struct Environment<'a> {
    available_values: &'a HashMap<SkyKey, Arc<dyn SkyValue>>,
    requested_deps: Vec<SkyKey>,
    values_missing: bool,
}

impl<'a> Environment<'a> {
    pub fn new(available_values: &'a HashMap<SkyKey, Arc<dyn SkyValue>>) -> Self {
        Self {
            available_values,
            requested_deps: Vec::new(),
            values_missing: false,
        }
    }

    pub fn get_value(&mut self, key: &SkyKey) -> Option<Arc<dyn SkyValue>> {
        if !self.requested_deps.contains(key) {
            self.requested_deps.push(key.clone());
        }
        match self.available_values.get(key) {
            Some(v) => Some(Arc::clone(v)),
            None => {
                self.values_missing = true;
                None
            }
        }
    }

    pub fn values_missing(&self) -> bool {
        self.values_missing
    }

    pub fn into_requested_deps(self) -> Vec<SkyKey> {
        self.requested_deps
    }
}

/// Core `SkyFunction` trait matching `com.google.devtools.build.skyframe.SkyFunction`.
pub trait SkyFunction: Send + Sync {
    /// Computes the `SkyValue` for `key`. Returns `Ok(None)` when `env.values_missing()` is true.
    fn compute(
        &self,
        key: &SkyKey,
        env: &mut Environment<'_>,
    ) -> Result<Option<Arc<dyn SkyValue>>, SkyFunctionError>;
}

#[derive(Debug, Clone)]
pub struct NodeEntry {
    pub value: Arc<dyn SkyValue>,
    pub direct_deps: Vec<SkyKey>,
    pub reverse_deps: HashSet<SkyKey>,
    pub dirty: bool,
}

#[derive(Debug)]
pub struct EvaluationResult {
    pub values: HashMap<SkyKey, Arc<dyn SkyValue>>,
    pub errors: HashMap<SkyKey, SkyFunctionError>,
    pub cycles: Vec<Vec<SkyKey>>,
}

impl EvaluationResult {
    pub fn has_error(&self) -> bool {
        !self.errors.is_empty() || !self.cycles.is_empty()
    }

    pub fn get<T: SkyValue>(&self, key: &SkyKey) -> Option<&T> {
        self.values.get(key)?.as_any().downcast_ref::<T>()
    }
}

/// Incremental memoizing evaluator matching `InMemoryMemoizingEvaluator`.
pub struct InMemoryMemoizingEvaluator {
    functions: HashMap<SkyFunctionName, Arc<dyn SkyFunction>>,
    graph: RwLock<HashMap<SkyKey, NodeEntry>>,
}

impl Default for InMemoryMemoizingEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryMemoizingEvaluator {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            graph: RwLock::new(HashMap::new()),
        }
    }

    pub fn register_function(
        &mut self,
        name: SkyFunctionName,
        func: Arc<dyn SkyFunction>,
    ) -> &mut Self {
        self.functions.insert(name, func);
        self
    }

    pub fn inject_precomputed(&self, key: SkyKey, value: Arc<dyn SkyValue>) {
        let mut graph = self.graph.write().unwrap();
        let mut rdeps_to_dirty = None;
        if let Some(existing) = graph.get_mut(&key) {
            if !existing.value.eq_value(value.as_ref()) {
                existing.value = value;
                rdeps_to_dirty = Some(existing.reverse_deps.iter().cloned().collect::<Vec<SkyKey>>());
            }
            existing.dirty = false;
        } else {
            graph.insert(
                key,
                NodeEntry {
                    value,
                    direct_deps: Vec::new(),
                    reverse_deps: HashSet::new(),
                    dirty: false,
                },
            );
        }
        if let Some(rdeps) = rdeps_to_dirty {
            Self::mark_dirty_recursive(&mut graph, &rdeps);
        }
    }

    /// Marks `keys` and their transitive reverse dependencies as dirty in the Skyframe graph.
    pub fn invalidate(&self, keys: &[SkyKey]) {
        let mut graph = self.graph.write().unwrap();
        Self::mark_dirty_recursive(&mut graph, keys);
    }

    fn mark_dirty_recursive(graph: &mut HashMap<SkyKey, NodeEntry>, keys: &[SkyKey]) {
        let mut stack: Vec<SkyKey> = keys.to_vec();
        let mut visited = HashSet::new();
        while let Some(k) = stack.pop() {
            if !visited.insert(k.clone()) {
                continue;
            }
            if let Some(entry) = graph.get_mut(&k) {
                entry.dirty = true;
                for rdep in &entry.reverse_deps {
                    stack.push(rdep.clone());
                }
            }
        }
    }

    pub fn evaluate(&self, roots: &[SkyKey]) -> EvaluationResult {
        let mut result = EvaluationResult {
            values: HashMap::new(),
            errors: HashMap::new(),
            cycles: Vec::new(),
        };
        let mut visiting = Vec::new();
        for root in roots {
            let _ = self.evaluate_key(root, &mut visiting, &mut result);
        }
        result
    }

    fn evaluate_key(
        &self,
        key: &SkyKey,
        visiting: &mut Vec<SkyKey>,
        result: &mut EvaluationResult,
    ) -> Option<Arc<dyn SkyValue>> {
        if let Some(pos) = visiting.iter().position(|k| k == key) {
            let mut cycle = visiting[pos..].to_vec();
            cycle.push(key.clone());
            result.cycles.push(cycle);
            return None;
        }

        // Fast path: clean node already in graph
        {
            let graph = self.graph.read().unwrap();
            if let Some(entry) = graph.get(key) {
                if !entry.dirty {
                    result.values.insert(key.clone(), Arc::clone(&entry.value));
                    return Some(Arc::clone(&entry.value));
                }
            }
        }

        let func = match self.functions.get(&key.function_name()) {
            Some(f) => Arc::clone(f),
            None => {
                result.errors.insert(
                    key.clone(),
                    SkyFunctionError::persistent(format!(
                        "No SkyFunction registered for {}",
                        key.function_name()
                    )),
                );
                return None;
            }
        };

        visiting.push(key.clone());
        let mut available: HashMap<SkyKey, Arc<dyn SkyValue>> = HashMap::new();

        loop {
            let mut env = Environment::new(&available);
            match func.compute(key, &mut env) {
                Ok(Some(val)) => {
                    let deps = env.into_requested_deps();
                    visiting.pop();
                    let mut graph = self.graph.write().unwrap();
                    for dep in &deps {
                        if let Some(dep_entry) = graph.get_mut(dep) {
                            dep_entry.reverse_deps.insert(key.clone());
                        }
                    }
                    let existing_rdeps = graph
                        .get(key)
                        .map(|e| e.reverse_deps.clone())
                        .unwrap_or_default();
                    graph.insert(
                        key.clone(),
                        NodeEntry {
                            value: Arc::clone(&val),
                            direct_deps: deps,
                            reverse_deps: existing_rdeps,
                            dirty: false,
                        },
                    );
                    result.values.insert(key.clone(), Arc::clone(&val));
                    return Some(val);
                }
                Ok(None) => {
                    let deps = env.into_requested_deps();
                    let mut any_failed = false;
                    for dep in deps {
                        if available.contains_key(&dep) {
                            continue;
                        }
                        if let Some(dep_val) = self.evaluate_key(&dep, visiting, result) {
                            available.insert(dep, dep_val);
                        } else {
                            any_failed = true;
                        }
                    }
                    if any_failed {
                        visiting.pop();
                        return None;
                    }
                }
                Err(err) => {
                    visiting.pop();
                    result.errors.insert(key.clone(), err);
                    return None;
                }
            }
        }
    }
}

/// Built-in `FILE_STATE` SkyFunction backed by `bazel_vfs::FileSystem`.
pub struct FileStateFunction {
    pub fs: Arc<dyn FileSystem>,
}

impl SkyFunction for FileStateFunction {
    fn compute(
        &self,
        key: &SkyKey,
        _env: &mut Environment<'_>,
    ) -> Result<Option<Arc<dyn SkyValue>>, SkyFunctionError> {
        let SkyKey::FileState(rooted_path) = key else {
            return Err(SkyFunctionError::persistent("Expected FileState SkyKey"));
        };
        let full_path = rooted_path.as_path();
        match self.fs.stat(&full_path, false) {
            Ok(st) => {
                if st.is_symlink {
                    let target = self
                        .fs
                        .read_symbolic_link(&full_path)
                        .map_err(|e| SkyFunctionError::transient(e.to_string()))?;
                    Ok(Some(Arc::new(FileStateValue::Symlink { target })))
                } else if st.is_directory {
                    Ok(Some(Arc::new(FileStateValue::Directory)))
                } else {
                    let digest = self
                        .fs
                        .get_digest(&full_path)
                        .map_err(|e| SkyFunctionError::transient(e.to_string()))?;
                    Ok(Some(Arc::new(FileStateValue::RegularFile {
                        size: st.size,
                        digest,
                    })))
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Ok(Some(Arc::new(FileStateValue::NonExistent)))
            }
            Err(e) => Err(SkyFunctionError::transient(e.to_string())),
        }
    }
}

/// Built-in `PACKAGE_LOOKUP` SkyFunction that checks `BUILD.bazel` then `BUILD`.
pub struct PackageLookupFunction {
    pub workspace_root: PathFragment,
}

impl SkyFunction for PackageLookupFunction {
    fn compute(
        &self,
        key: &SkyKey,
        env: &mut Environment<'_>,
    ) -> Result<Option<Arc<dyn SkyValue>>, SkyFunctionError> {
        let SkyKey::PackageLookup(pkg_id) = key else {
            return Err(SkyFunctionError::persistent(
                "Expected PackageLookup SkyKey",
            ));
        };
        let pkg_frag = pkg_id.package_fragment();
        let build_bazel_rp = RootedPath::to_rooted_path(
            self.workspace_root.clone(),
            pkg_frag.get_relative_str("BUILD.bazel"),
        )
        .map_err(SkyFunctionError::persistent)?;
        let build_rp = RootedPath::to_rooted_path(
            self.workspace_root.clone(),
            pkg_frag.get_relative_str("BUILD"),
        )
        .map_err(SkyFunctionError::persistent)?;

        let v1 = env.get_value(&SkyKey::FileState(build_bazel_rp.clone()));
        let v2 = env.get_value(&SkyKey::FileState(build_rp.clone()));
        if env.values_missing() {
            return Ok(None);
        }

        if let Some(fs1) = v1.and_then(|v| v.as_any().downcast_ref::<FileStateValue>().cloned()) {
            if matches!(fs1, FileStateValue::RegularFile { .. }) {
                return Ok(Some(Arc::new(PackageLookupValue {
                    package_exists: true,
                    build_file_path: Some(build_bazel_rp),
                })));
            }
        }
        if let Some(fs2) = v2.and_then(|v| v.as_any().downcast_ref::<FileStateValue>().cloned()) {
            if matches!(fs2, FileStateValue::RegularFile { .. }) {
                return Ok(Some(Arc::new(PackageLookupValue {
                    package_exists: true,
                    build_file_path: Some(build_rp),
                })));
            }
        }

        Ok(Some(Arc::new(PackageLookupValue {
            package_exists: false,
            build_file_path: None,
        })))
    }
}

/// Built-in `PACKAGE` SkyFunction that loads and evaluates a package's `BUILD` / `BUILD.bazel` file.
pub struct PackageFunction {
    pub fs: Arc<dyn FileSystem>,
}

impl SkyFunction for PackageFunction {
    fn compute(
        &self,
        key: &SkyKey,
        env: &mut Environment<'_>,
    ) -> Result<Option<Arc<dyn SkyValue>>, SkyFunctionError> {
        let SkyKey::Package(pkg_id) = key else {
            return Err(SkyFunctionError::persistent("Expected Package SkyKey"));
        };
        let Some(lookup_val) = env.get_value(&SkyKey::PackageLookup(pkg_id.clone())) else {
            return Ok(None);
        };
        let lookup = lookup_val
            .as_any()
            .downcast_ref::<PackageLookupValue>()
            .ok_or_else(|| SkyFunctionError::persistent("Invalid PackageLookupValue"))?;

        if !lookup.package_exists {
            return Err(SkyFunctionError::persistent(format!(
                "no such package '{}': BUILD file not found",
                pkg_id.package_fragment()
            )));
        }
        let build_rp = lookup.build_file_path.clone().unwrap();
        let _ = env.get_value(&SkyKey::FileState(build_rp.clone()));
        if env.values_missing() {
            return Ok(None);
        }

        let content_bytes = self
            .fs
            .read_file(&build_rp.as_path())
            .map_err(|e| SkyFunctionError::transient(e.to_string()))?;
        let content = String::from_utf8_lossy(&content_bytes);

        let workspace_root = build_rp.root().clone();
        let (targets, package_opt) = match Package::evaluate_build_file(
            pkg_id.clone(),
            &workspace_root,
            build_rp.as_path(),
            &content,
            Arc::clone(&self.fs),
        ) {
            Ok(pkg) => {
                let mut map = BTreeMap::new();
                for rule in pkg.rules() {
                    map.insert(
                        rule.label.name().to_string(),
                        RuleTarget {
                            label: rule.label.clone(),
                            rule_class: rule.rule_class.clone(),
                            srcs: rule.srcs.clone(),
                            deps: rule.deps.clone(),
                            cmd: rule.cmd.clone(),
                            outs: rule.outs.clone(),
                        },
                    );
                }
                if map.is_empty() {
                    if let Ok(fallback) = parse_build_targets(pkg_id, &content) {
                        map = fallback;
                    }
                }
                (map, Some(pkg))
            }
            Err(_) => (parse_build_targets(pkg_id, &content)?, None),
        };

        Ok(Some(Arc::new(PackageValue {
            package_id: pkg_id.clone(),
            build_file: build_rp,
            targets,
            package: package_opt,
        })))
    }
}

/// Built-in `CONFIGURED_TARGET` SkyFunction that analyzes a target in a `PackageValue`.
pub struct ConfiguredTargetFunction {
    pub default_config: BuildConfigurationKey,
}

impl SkyFunction for ConfiguredTargetFunction {
    fn compute(
        &self,
        key: &SkyKey,
        env: &mut Environment<'_>,
    ) -> Result<Option<Arc<dyn SkyValue>>, SkyFunctionError> {
        let SkyKey::ConfiguredTarget { label, .. } = key else {
            return Err(SkyFunctionError::persistent(
                "Expected ConfiguredTarget SkyKey",
            ));
        };
        let pkg_key = SkyKey::Package(label.package_identifier().clone());
        let Some(pkg_val_any) = env.get_value(&pkg_key) else {
            return Ok(None);
        };
        let pkg_val = pkg_val_any
            .as_any()
            .downcast_ref::<PackageValue>()
            .ok_or_else(|| SkyFunctionError::persistent("Invalid PackageValue"))?;

        let target = if let Some(ref pkg) = pkg_val.package {
            if let Some(t) = pkg.targets.get(label.name()) {
                t.clone()
            } else {
                Target::InputFile {
                    label: label.clone(),
                }
            }
        } else if let Some(rt) = pkg_val.targets.get(label.name()) {
            Target::Rule(bazel_packages::RuleInstance {
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
        } else {
            Target::InputFile {
                label: label.clone(),
            }
        };

        let ct = analyze_target(&target, &self.default_config)
            .map_err(SkyFunctionError::persistent)?;
        Ok(Some(Arc::new(ConfiguredTargetValue {
            configured_target: ct,
        })))
    }
}

/// Parses Starlark rule instantiations in a `BUILD` file (`cc_binary`, `cc_library`, `java_library`,
/// `java_binary`, `rust_library`, `rust_binary`, `genrule`, `filegroup`, `sh_binary`, `sh_test`, etc.).
pub fn parse_build_targets(
    pkg_id: &PackageIdentifier,
    build_content: &str,
) -> Result<BTreeMap<String, RuleTarget>, SkyFunctionError> {
    let mut targets = BTreeMap::new();
    let mut pos = 0;
    let bytes = build_content.as_bytes();

    while pos < bytes.len() {
        // Skip whitespace and comments
        if bytes[pos].is_ascii_whitespace() {
            pos += 1;
            continue;
        }
        if bytes[pos] == b'#' {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        if bytes[pos].is_ascii_alphabetic() || bytes[pos] == b'_' {
            let ident_start = pos;
            while pos < bytes.len()
                && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_')
            {
                pos += 1;
            }
            let ident = &build_content[ident_start..pos];
            while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\t') {
                pos += 1;
            }
            if pos < bytes.len() && bytes[pos] == b'(' {
                let args_start = pos + 1;
                let mut depth = 1usize;
                pos += 1;
                let mut in_str: Option<u8> = None;
                while pos < bytes.len() && depth > 0 {
                    let b = bytes[pos];
                    if let Some(q) = in_str {
                        if b == b'\\' {
                            pos += 2;
                            continue;
                        } else if b == q {
                            in_str = None;
                        }
                    } else if b == b'"' || b == b'\'' {
                        in_str = Some(b);
                    } else if b == b'#' {
                        while pos < bytes.len() && bytes[pos] != b'\n' {
                            pos += 1;
                        }
                        continue;
                    } else if b == b'(' {
                        depth += 1;
                    } else if b == b')' {
                        depth -= 1;
                    }
                    pos += 1;
                }
                let args_body = &build_content[args_start..pos.saturating_sub(1)];
                if !matches!(
                    ident,
                    "load" | "package" | "licenses" | "exports_files" | "select" | "glob"
                ) {
                    if let Some(name) = extract_string_kwarg(args_body, "name") {
                        if let Ok(label) = Label::create(pkg_id.clone(), &name) {
                            let srcs = extract_list_kwarg(args_body, "srcs");
                            let deps = extract_list_kwarg(args_body, "deps");
                            let outs = extract_list_kwarg(args_body, "outs");
                            let cmd = extract_string_kwarg(args_body, "cmd");
                            targets.insert(
                                name,
                                RuleTarget {
                                    label,
                                    rule_class: ident.to_string(),
                                    srcs,
                                    deps,
                                    cmd,
                                    outs,
                                },
                            );
                        }
                    }
                }
                continue;
            }
        }
        pos += 1;
    }

    Ok(targets)
}

fn extract_string_kwarg(body: &str, key: &str) -> Option<String> {
    for part in body.lines() {
        let trimmed = part.trim();
        if let Some(rest) = trimmed.strip_prefix(key) {
            let rest = rest.trim_start();
            if let Some(after_eq) = rest.strip_prefix('=') {
                let after_eq = after_eq.trim_start();
                let quote = after_eq.chars().next()?;
                if quote == '"' || quote == '\'' {
                    let after_q = &after_eq[1..];
                    let end_idx = after_q.find(quote)?;
                    return Some(after_q[..end_idx].to_string());
                }
            }
        }
    }
    None
}

fn extract_list_kwarg(body: &str, key: &str) -> Vec<String> {
    let pattern = format!("{key}");
    let mut search_from = 0;
    while let Some(rel_idx) = body[search_from..].find(&pattern) {
        let idx = search_from + rel_idx;
        let before_ok = idx == 0
            || !body.as_bytes()[idx - 1].is_ascii_alphanumeric()
                && body.as_bytes()[idx - 1] != b'_';
        let after = &body[idx + pattern.len()..];
        let after_trimmed = after.trim_start();
        if before_ok {
            if let Some(after_eq) = after_trimmed.strip_prefix('=') {
                let after_eq = after_eq.trim_start();
                if let Some(after_bracket) = after_eq.strip_prefix('[') {
                    if let Some(end_bracket) = after_bracket.find(']') {
                        let list_content = &after_bracket[..end_bracket];
                        let mut items = Vec::new();
                        let mut chars = list_content.chars();
                        while let Some(c) = chars.next() {
                            if c == '"' || c == '\'' {
                                let mut s = String::new();
                                for nc in chars.by_ref() {
                                    if nc == c {
                                        break;
                                    }
                                    s.push(nc);
                                }
                                items.push(s);
                            }
                        }
                        return items;
                    }
                }
            }
        }
        search_from = idx + pattern.len();
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bazel_vfs::InMemoryFileSystem;

    #[test]
    fn test_skyframe_package_loading_and_invalidation() {
        let fs = Arc::new(InMemoryFileSystem::default());
        let ws = PathFragment::create("/workspace");
        let pkg_dir = ws.get_relative_str("foo");
        fs.create_directory_and_parents(&pkg_dir).unwrap();
        let build_file = pkg_dir.get_relative_str("BUILD");
        fs.write_file(
            &build_file,
            br#"
cc_library(
    name = "lib",
    srcs = ["lib.cc"],
)
cc_binary(
    name = "bin",
    srcs = ["main.cc"],
    deps = [":lib"],
)
"#,
        )
        .unwrap();

        let mut evaluator = InMemoryMemoizingEvaluator::new();
        evaluator
            .register_function(
                SkyFunctionName::FILE_STATE,
                Arc::new(FileStateFunction {
                    fs: Arc::clone(&fs) as Arc<dyn FileSystem>,
                }),
            )
            .register_function(
                SkyFunctionName::PACKAGE_LOOKUP,
                Arc::new(PackageLookupFunction {
                    workspace_root: ws.clone(),
                }),
            )
            .register_function(
                SkyFunctionName::PACKAGE,
                Arc::new(PackageFunction {
                    fs: Arc::clone(&fs) as Arc<dyn FileSystem>,
                }),
            );

        let pkg_id = PackageIdentifier::create_in_main_repo("foo").unwrap();
        let pkg_key = SkyKey::Package(pkg_id.clone());
        let res = evaluator.evaluate(std::slice::from_ref(&pkg_key));
        assert!(!res.has_error());

        let pkg_val = res.get::<PackageValue>(&pkg_key).unwrap();
        assert_eq!(pkg_val.targets.len(), 2);
        assert_eq!(pkg_val.targets["bin"].rule_class, "cc_binary");
        assert_eq!(pkg_val.targets["bin"].deps, vec![":lib"]);

        // Now modify the BUILD file and invalidate its FileState node
        fs.write_file(
            &build_file,
            br#"
cc_binary(
    name = "new_bin",
    srcs = ["main.cc"],
)
"#,
        )
        .unwrap();
        let build_rp =
            RootedPath::to_rooted_path(ws.clone(), PathFragment::create("foo/BUILD")).unwrap();
        evaluator.invalidate(&[SkyKey::FileState(build_rp)]);

        let res2 = evaluator.evaluate(std::slice::from_ref(&pkg_key));
        let pkg_val2 = res2.get::<PackageValue>(&pkg_key).unwrap();
        assert_eq!(pkg_val2.targets.len(), 1);
        assert!(pkg_val2.targets.contains_key("new_bin"));
    }
}
