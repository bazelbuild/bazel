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

//! Bazel Label, PackageIdentifier, RepositoryName, RepositoryMapping, and TargetPattern parsing.
//! Mirrors `src/main/java/com/google/devtools/build/lib/cmdline`.

use bazel_vfs::PathFragment;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelSyntaxError(pub String);

impl fmt::Display for LabelSyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LabelSyntaxError {}

/// Exact port of `com.google.devtools.build.lib.cmdline.LabelValidator`.
pub mod validator {
    pub const PACKAGE_NAME_ERROR: &str =
        "package names may contain A-Z, a-z, 0-9, or any of ' !\"#$%&'()*+,-./;<=>?[]^_`{|}~' (any ASCII character except 0-31, 127, ':', or '\\')";
    pub const PACKAGE_NAME_DOT_ERROR: &str = "package name component contains only '.' characters";

    fn is_allowed_in_package_name(c: char) -> bool {
        c.is_ascii_alphanumeric() || " !\"#$%&'()*+,-./;<=>?@[]^_`{|}~".contains(c)
    }

    fn is_always_allowed_in_target(c: char) -> bool {
        if !c.is_ascii() {
            return true;
        }
        c.is_ascii_alphanumeric()
            || " \"#$&'()*+,;<=>?[]{|}~".contains(c)
            || "!%-@^_`".contains(c)
    }

    pub fn validate_package_name(package_name: &str) -> Result<(), &'static str> {
        if package_name.is_empty() {
            return Ok(());
        }
        if package_name.starts_with('/') {
            return Err("package names may not start with '/'");
        }
        if !package_name.chars().all(is_allowed_in_package_name) {
            return Err(PACKAGE_NAME_ERROR);
        }
        if package_name.ends_with('/') {
            return Err("package names may not end with '/'");
        }

        let bytes = package_name.as_bytes();
        let mut non_dot = false;
        let mut last_slash = true;
        for i in (-1..(bytes.len() as isize)).rev() {
            let c = if i >= 0 { bytes[i as usize] } else { b'/' };
            if c == b'/' {
                if last_slash {
                    return Err("package names may not contain '//' path separators");
                }
                if !non_dot {
                    return Err(PACKAGE_NAME_DOT_ERROR);
                }
                non_dot = false;
                last_slash = true;
            } else {
                if c != b'.' {
                    non_dot = true;
                }
                last_slash = false;
            }
        }
        Ok(())
    }

    pub fn validate_target_name(target_name: &str) -> Result<(), String> {
        if target_name.is_empty() {
            return Err("empty target name".to_string());
        }
        if target_name.starts_with('/') {
            return Err("target names may not start with '/'".to_string());
        }
        if target_name.starts_with('.') {
            if target_name.starts_with("../") || target_name == ".." {
                return Err("target names may not contain up-level references '..'".to_string());
            } else if target_name == "." {
                return Ok(());
            } else if target_name.starts_with("./") {
                return Err("target names may not contain '.' as a path segment".to_string());
            }
        }
        if target_name.ends_with('\r') {
            return Err(
                "target names may not end with carriage returns (perhaps the input source is CRLF-terminated)"
                    .to_string(),
            );
        }

        let bytes = target_name.as_bytes();
        for (ii, c) in target_name.char_indices() {
            if is_always_allowed_in_target(c) || c == '.' {
                continue;
            }
            if c == '/' {
                if target_name[ii..].starts_with("/../") {
                    return Err("target names may not contain up-level references '..'".to_string());
                } else if target_name[ii..].starts_with("/./") {
                    return Err("target names may not contain '.' as a path segment".to_string());
                } else if target_name[ii..].starts_with("//") {
                    return Err("target names may not contain '//' path separators".to_string());
                }
                continue;
            }
            let b = bytes[ii];
            if b <= 0x1f || b == 0x7f {
                return Err(format!(
                    "target names may not contain non-printable characters: '\\x{:02X}'",
                    b
                ));
            }
            return Err(format!("target names may not contain '{c}'"));
        }

        if target_name.ends_with("/..") {
            return Err("target names may not contain up-level references '..'".to_string());
        }
        if target_name.ends_with("/.") {
            return Ok(());
        }
        if target_name.ends_with('/') {
            return Err("target names may not end with '/'".to_string());
        }
        Ok(())
    }
}

/// The canonical name of an external repository, matching `RepositoryName.java`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct RepositoryName {
    name: String,
}

impl RepositoryName {
    pub fn main() -> Self {
        Self {
            name: String::new(),
        }
    }

    pub fn bazel_tools() -> Self {
        Self {
            name: "bazel_tools".to_string(),
        }
    }

    pub fn builtins() -> Self {
        Self {
            name: "_builtins".to_string(),
        }
    }

    pub fn create(name: &str) -> Result<Self, LabelSyntaxError> {
        Self::validate(name)?;
        Ok(Self {
            name: name.to_string(),
        })
    }

    pub fn create_unvalidated(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn validate(name: &str) -> Result<(), LabelSyntaxError> {
        if name.is_empty() || name == "_builtins" {
            return Ok(());
        }
        if name == "." || name == ".." {
            return Err(LabelSyntaxError(format!(
                "invalid repository name '{name}': repo names are not allowed to be '{name}'"
            )));
        }
        let valid = name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '+'));
        if !valid {
            return Err(LabelSyntaxError(format!(
                "invalid repository name '{name}': repo names may contain only A-Z, a-z, 0-9, '-', '_', '.' and '+'"
            )));
        }
        Ok(())
    }

    pub fn is_apparent(name: &str) -> bool {
        !name.is_empty() && !name.contains('+')
    }

    pub fn is_main(&self) -> bool {
        self.name.is_empty()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn name_with_at(&self) -> String {
        format!("@@{}", self.name)
    }

    pub fn canonical_form(&self) -> String {
        if self.is_main() {
            String::new()
        } else {
            self.name_with_at()
        }
    }
}

impl fmt::Display for RepositoryName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.canonical_form())
    }
}

/// Maps apparent repository names (`@foo`) to canonical repository names (`@@foo+1.0`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RepositoryMapping {
    entries: HashMap<String, RepositoryName>,
    owner_repo: RepositoryName,
}

impl RepositoryMapping {
    pub fn always_fallback() -> Self {
        Self::default()
    }

    pub fn new(entries: HashMap<String, RepositoryName>, owner_repo: RepositoryName) -> Self {
        Self {
            entries,
            owner_repo,
        }
    }

    pub fn get(&self, apparent_name: &str) -> RepositoryName {
        if apparent_name.is_empty() {
            return self.owner_repo.clone();
        }
        self.entries
            .get(apparent_name)
            .cloned()
            .unwrap_or_else(|| RepositoryName::create_unvalidated(apparent_name))
    }
}

/// Identifies a Bazel package (`RepositoryName` + package `PathFragment`), matching `PackageIdentifier.java`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackageIdentifier {
    repository: RepositoryName,
    pkg_fragment: PathFragment,
}

impl PackageIdentifier {
    pub fn create(
        repository: RepositoryName,
        pkg_fragment: PathFragment,
    ) -> Result<Self, LabelSyntaxError> {
        if pkg_fragment.is_absolute() {
            return Err(LabelSyntaxError(format!(
                "invalid package name '{}': package names may not start with '/'",
                pkg_fragment
            )));
        }
        if let Err(msg) = validator::validate_package_name(pkg_fragment.path_string()) {
            return Err(LabelSyntaxError(format!(
                "invalid package name '{}': {}",
                pkg_fragment, msg
            )));
        }
        Ok(Self {
            repository,
            pkg_fragment,
        })
    }

    pub fn create_in_main_repo(pkg: &str) -> Result<Self, LabelSyntaxError> {
        Self::create(RepositoryName::main(), PathFragment::create(pkg))
    }

    pub fn repository(&self) -> &RepositoryName {
        &self.repository
    }

    pub fn package_fragment(&self) -> &PathFragment {
        &self.pkg_fragment
    }

    pub fn canonical_form(&self) -> String {
        format!("{}//{}", self.repository.canonical_form(), self.pkg_fragment)
    }
}

impl fmt::Display for PackageIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.canonical_form())
    }
}

/// A parsed Bazel target label (`@@repo//pkg:target`), matching `Label.java`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Label {
    package_identifier: PackageIdentifier,
    name: String,
}

impl Label {
    pub fn create(pkg: PackageIdentifier, name: &str) -> Result<Self, LabelSyntaxError> {
        if let Err(msg) = validator::validate_target_name(name) {
            return Err(LabelSyntaxError(format!(
                "invalid target name '{name}': {msg}"
            )));
        }
        Ok(Self {
            package_identifier: pkg,
            name: name.to_string(),
        })
    }

    /// Parses an absolute label (`//pkg:target`, `@repo//pkg:target`, `@@repo//pkg`, `@repo`).
    pub fn parse_canonical(raw: &str) -> Result<Self, LabelSyntaxError> {
        Self::parse_with_repo_mapping(raw, &RepositoryMapping::always_fallback())
    }

    pub fn parse_with_repo_mapping(
        raw: &str,
        repo_mapping: &RepositoryMapping,
    ) -> Result<Self, LabelSyntaxError> {
        if !raw.starts_with("//") && !raw.starts_with('@') {
            return Err(LabelSyntaxError(format!(
                "invalid label '{raw}': absolute label must start with '//' or '@'"
            )));
        }

        let (repo, rest) = if let Some(after_at_at) = raw.strip_prefix("@@") {
            match after_at_at.find("//") {
                Some(idx) => {
                    let repo_str = &after_at_at[..idx];
                    (RepositoryName::create(repo_str)?, &after_at_at[idx..])
                }
                None => {
                    let repo = RepositoryName::create(after_at_at)?;
                    let target = if after_at_at.is_empty() {
                        return Err(LabelSyntaxError("empty repository shorthand".to_string()));
                    } else {
                        after_at_at
                    };
                    let pkg = PackageIdentifier::create(repo, PathFragment::empty())?;
                    return Self::create(pkg, target);
                }
            }
        } else if let Some(after_at) = raw.strip_prefix('@') {
            match after_at.find("//") {
                Some(idx) => {
                    let apparent = &after_at[..idx];
                    RepositoryName::validate(apparent)?;
                    (repo_mapping.get(apparent), &after_at[idx..])
                }
                None => {
                    RepositoryName::validate(after_at)?;
                    let mapped = repo_mapping.get(after_at);
                    let target = if after_at.is_empty() {
                        return Err(LabelSyntaxError("empty repository shorthand".to_string()));
                    } else {
                        after_at
                    };
                    let pkg = PackageIdentifier::create(mapped, PathFragment::empty())?;
                    return Self::create(pkg, target);
                }
            }
        } else {
            (repo_mapping.get(""), raw)
        };

        let after_slashes = rest.strip_prefix("//").ok_or_else(|| {
            LabelSyntaxError(format!("invalid label '{raw}': missing '//' separator"))
        })?;

        let (pkg_str, target_str) = match after_slashes.find(':') {
            Some(colon_idx) => (&after_slashes[..colon_idx], &after_slashes[colon_idx + 1..]),
            None => {
                let default_target = match after_slashes.rfind('/') {
                    Some(slash_idx) => &after_slashes[slash_idx + 1..],
                    None => after_slashes,
                };
                (after_slashes, default_target)
            }
        };

        if let Err(err) = validator::validate_package_name(pkg_str) {
            let mut msg = format!("invalid package name '{pkg_str}': {err}");
            if pkg_str.ends_with(&format!("/{target_str}")) {
                msg.push_str(&format!(" (perhaps you meant \":{target_str}\"?)"));
            }
            return Err(LabelSyntaxError(msg));
        }

        let pkg = PackageIdentifier::create(repo, PathFragment::create(pkg_str))?;
        Self::create(pkg, target_str)
    }

    pub fn package_identifier(&self) -> &PackageIdentifier {
        &self.package_identifier
    }

    pub fn package_name(&self) -> &str {
        self.package_identifier.package_fragment().path_string()
    }

    pub fn repository(&self) -> &RepositoryName {
        self.package_identifier.repository()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn canonical_form(&self) -> String {
        format!("{}:{}", self.package_identifier.canonical_form(), self.name)
    }

    pub fn shorthand_form(&self) -> String {
        let pkg = self.package_name();
        let last_seg = pkg.rsplit('/').next().unwrap_or(pkg);
        if !pkg.is_empty() && last_seg == self.name {
            self.package_identifier.canonical_form()
        } else {
            self.canonical_form()
        }
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.canonical_form())
    }
}

/// Target pattern kinds matching `com.google.devtools.build.lib.cmdline.TargetPattern`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetPattern {
    /// A single target, e.g. `//foo/bar:baz` or `foo/bar`.
    SingleTarget(Label),
    /// All targets in a package, e.g. `//foo/bar:all` or `//foo/bar:*`.
    TargetsInPackage {
        package: PackageIdentifier,
        include_files: bool,
    },
    /// All targets recursively below a directory, e.g. `//foo/...` or `//foo/...:all`.
    TargetsBelowDirectory {
        directory: PackageIdentifier,
        include_files: bool,
    },
}

impl TargetPattern {
    pub fn parse(raw: &str, relative_dir: &PathFragment) -> Result<Self, LabelSyntaxError> {
        let normalized = if raw == "..." || raw.starts_with("...:") {
            if relative_dir.is_empty() {
                format!("//{raw}")
            } else {
                format!("//{relative_dir}/{raw}")
            }
        } else if !raw.starts_with("//") && !raw.starts_with('@') {
            if let Some(stripped) = raw.strip_prefix(':') {
                format!("//{relative_dir}:{stripped}")
            } else if relative_dir.is_empty() {
                format!("//{raw}")
            } else {
                format!("//{relative_dir}/{raw}")
            }
        } else {
            raw.to_string()
        };

        // Check for `...` recursive wildcard
        if let Some(base) = normalized
            .strip_suffix("/...:all")
            .or_else(|| normalized.strip_suffix("/..."))
        {
            let dir_str = if base == "//" || base.is_empty() {
                "//"
            } else {
                base
            };
            let pkg_str = dir_str.strip_prefix("//").unwrap_or("");
            let pkg = PackageIdentifier::create_in_main_repo(pkg_str)?;
            return Ok(Self::TargetsBelowDirectory {
                directory: pkg,
                include_files: false,
            });
        }
        if let Some(base) = normalized
            .strip_suffix("/...:*")
            .or_else(|| normalized.strip_suffix("/...:all-targets"))
        {
            let pkg_str = base.strip_prefix("//").unwrap_or("");
            let pkg = PackageIdentifier::create_in_main_repo(pkg_str)?;
            return Ok(Self::TargetsBelowDirectory {
                directory: pkg,
                include_files: true,
            });
        }
        if normalized == "//..." || normalized == "//...:all" {
            return Ok(Self::TargetsBelowDirectory {
                directory: PackageIdentifier::create_in_main_repo("")?,
                include_files: false,
            });
        }

        // Check for `:all`, `:*`, `:all-targets`
        if let Some(pkg_part) = normalized.strip_suffix(":all") {
            let pkg_str = pkg_part.strip_prefix("//").unwrap_or(pkg_part);
            return Ok(Self::TargetsInPackage {
                package: PackageIdentifier::create_in_main_repo(pkg_str)?,
                include_files: false,
            });
        }
        if let Some(pkg_part) = normalized
            .strip_suffix(":*")
            .or_else(|| normalized.strip_suffix(":all-targets"))
        {
            let pkg_str = pkg_part.strip_prefix("//").unwrap_or(pkg_part);
            return Ok(Self::TargetsInPackage {
                package: PackageIdentifier::create_in_main_repo(pkg_str)?,
                include_files: true,
            });
        }

        let label = Label::parse_canonical(&normalized)?;
        Ok(Self::SingleTarget(label))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternSign {
    Positive,
    Negative,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedTargetPattern {
    pub sign: PatternSign,
    pub pattern: TargetPattern,
}

impl SignedTargetPattern {
    pub fn parse(raw: &str, relative_dir: &PathFragment) -> Result<Self, LabelSyntaxError> {
        if let Some(stripped) = raw.strip_prefix('-') {
            Ok(Self {
                sign: PatternSign::Negative,
                pattern: TargetPattern::parse(stripped, relative_dir)?,
            })
        } else {
            let stripped = raw.strip_prefix('+').unwrap_or(raw);
            Ok(Self {
                sign: PatternSign::Positive,
                pattern: TargetPattern::parse(stripped, relative_dir)?,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_parsing_and_shorthand() {
        let l1 = Label::parse_canonical("//src/main/java:bazel").unwrap();
        assert_eq!(l1.package_name(), "src/main/java");
        assert_eq!(l1.name(), "bazel");
        assert_eq!(l1.canonical_form(), "//src/main/java:bazel");

        let l2 = Label::parse_canonical("//src/tools/singlejar").unwrap();
        assert_eq!(l2.package_name(), "src/tools/singlejar");
        assert_eq!(l2.name(), "singlejar");
        assert_eq!(l2.canonical_form(), "//src/tools/singlejar:singlejar");
        assert_eq!(l2.shorthand_form(), "//src/tools/singlejar");

        let l3 = Label::parse_canonical("@rules_rs//rs:rust_binary.bzl").unwrap();
        assert_eq!(l3.repository().name(), "rules_rs");
        assert_eq!(l3.package_name(), "rs");
        assert_eq!(l3.name(), "rust_binary.bzl");
        assert_eq!(l3.canonical_form(), "@@rules_rs//rs:rust_binary.bzl");
    }

    #[test]
    fn test_label_validation_errors() {
        assert!(Label::parse_canonical("//foo//bar:baz").is_err());
        assert!(Label::parse_canonical("//foo/..:baz").is_err());
        assert!(Label::parse_canonical("//foo:../baz").is_err());
        assert!(Label::parse_canonical("//foo:bar/").is_err());
    }

    #[test]
    fn test_target_patterns() {
        let rel = PathFragment::empty();
        let p1 = TargetPattern::parse("//src/test/...", &rel).unwrap();
        assert_eq!(
            p1,
            TargetPattern::TargetsBelowDirectory {
                directory: PackageIdentifier::create_in_main_repo("src/test").unwrap(),
                include_files: false,
            }
        );

        let p2 = SignedTargetPattern::parse("-//src/test/shell:all", &rel).unwrap();
        assert_eq!(p2.sign, PatternSign::Negative);
    }
}

