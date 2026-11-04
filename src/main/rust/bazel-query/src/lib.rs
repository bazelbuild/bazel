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

//! Bazel Query, Cquery, and Aquery AST, evaluator, and output formatters.
//! Mirrors `src/main/java/com/google/devtools/build/lib/query2`.

use bazel_cmdline::{Label, PackageIdentifier, TargetPattern};
use bazel_packages::{Package, Target};
use bazel_vfs::PathFragment;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryExpr {
    TargetPattern(String),
    Var(String),
    Deps {
        target: Box<QueryExpr>,
        max_depth: Option<usize>,
    },
    Rdeps {
        universe: Box<QueryExpr>,
        target: Box<QueryExpr>,
        max_depth: Option<usize>,
    },
    SomePath {
        from: Box<QueryExpr>,
        to: Box<QueryExpr>,
    },
    AllPaths {
        from: Box<QueryExpr>,
        to: Box<QueryExpr>,
    },
    Kind {
        pattern: String,
        input: Box<QueryExpr>,
    },
    Filter {
        pattern: String,
        input: Box<QueryExpr>,
    },
    Attr {
        attr_name: String,
        pattern: String,
        input: Box<QueryExpr>,
    },
    Tests(Box<QueryExpr>),
    Siblings(Box<QueryExpr>),
    BuildFiles(Box<QueryExpr>),
    Union(Box<QueryExpr>, Box<QueryExpr>),
    Except(Box<QueryExpr>, Box<QueryExpr>),
    Intersect(Box<QueryExpr>, Box<QueryExpr>),
    Let {
        var_name: String,
        val: Box<QueryExpr>,
        body: Box<QueryExpr>,
    },
}

impl QueryExpr {
    pub fn parse(raw: &str) -> Result<Self, String> {
        let tokens = tokenize_query(raw);
        let mut pos = 0;
        parse_query_union(&tokens, &mut pos)
    }
}

fn tokenize_query(raw: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = raw.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        if matches!(c, '(' | ')' | ',' | '+' | '^' | '=') {
            tokens.push(c.to_string());
            chars.next();
            continue;
        }
        if c == '"' || c == '\'' {
            let quote = chars.next().unwrap();
            let mut s = String::new();
            for nc in chars.by_ref() {
                if nc == quote {
                    break;
                }
                s.push(nc);
            }
            tokens.push(s);
            continue;
        }
        let mut word = String::new();
        while let Some(&nc) = chars.peek() {
            if nc.is_whitespace() || matches!(nc, '(' | ')' | ',' | '+' | '^' | '=') {
                break;
            }
            word.push(nc);
            chars.next();
        }
        tokens.push(word);
    }
    tokens
}

fn parse_query_union(tokens: &[String], pos: &mut usize) -> Result<QueryExpr, String> {
    if tokens.get(*pos).map(|s| s.as_str()) == Some("let") {
        *pos += 1;
        let var_name = tokens
            .get(*pos)
            .ok_or_else(|| "Expected variable name after 'let'".to_string())?
            .trim_start_matches('$')
            .to_string();
        *pos += 1;
        if tokens.get(*pos).map(|s| s.as_str()) == Some("=") {
            *pos += 1;
        }
        let val = parse_query_union(tokens, pos)?;
        if tokens.get(*pos).map(|s| s.as_str()) == Some("in") {
            *pos += 1;
        }
        let body = parse_query_union(tokens, pos)?;
        return Ok(QueryExpr::Let {
            var_name,
            val: Box::new(val),
            body: Box::new(body),
        });
    }

    let mut lhs = parse_query_primary(tokens, pos)?;
    while let Some(op) = tokens.get(*pos).map(|s| s.as_str()) {
        match op {
            "+" | "union" => {
                *pos += 1;
                let rhs = parse_query_primary(tokens, pos)?;
                lhs = QueryExpr::Union(Box::new(lhs), Box::new(rhs));
            }
            "-" | "except" => {
                *pos += 1;
                let rhs = parse_query_primary(tokens, pos)?;
                lhs = QueryExpr::Except(Box::new(lhs), Box::new(rhs));
            }
            "^" | "intersect" => {
                *pos += 1;
                let rhs = parse_query_primary(tokens, pos)?;
                lhs = QueryExpr::Intersect(Box::new(lhs), Box::new(rhs));
            }
            _ => break,
        }
    }
    Ok(lhs)
}

fn parse_query_primary(tokens: &[String], pos: &mut usize) -> Result<QueryExpr, String> {
    let tok = tokens
        .get(*pos)
        .ok_or_else(|| "Unexpected end of query expression".to_string())?
        .clone();
    *pos += 1;

    if tok == "(" {
        let inner = parse_query_union(tokens, pos)?;
        if tokens.get(*pos).map(|s| s.as_str()) == Some(")") {
            *pos += 1;
        }
        return Ok(inner);
    }

    if let Some(var_name) = tok.strip_prefix('$') {
        return Ok(QueryExpr::Var(var_name.to_string()));
    }

    if tokens.get(*pos).map(|s| s.as_str()) == Some("(") {
        *pos += 1; // eat '('
        let mut args_raw = Vec::new();
        while *pos < tokens.len() && tokens[*pos] != ")" {
            if tokens[*pos] == "," {
                *pos += 1;
                continue;
            }
            args_raw.push(parse_query_union(tokens, pos)?);
        }
        if tokens.get(*pos).map(|s| s.as_str()) == Some(")") {
            *pos += 1;
        }

        match tok.as_str() {
            "deps" => {
                let target = args_raw
                    .first()
                    .cloned()
                    .ok_or_else(|| "deps() requires at least 1 argument".to_string())?;
                let max_depth = args_raw.get(1).and_then(expr_as_usize);
                Ok(QueryExpr::Deps {
                    target: Box::new(target),
                    max_depth,
                })
            }
            "rdeps" => {
                if args_raw.len() < 2 {
                    return Err("rdeps() requires at least 2 arguments".to_string());
                }
                let max_depth = args_raw.get(2).and_then(expr_as_usize);
                Ok(QueryExpr::Rdeps {
                    universe: Box::new(args_raw[0].clone()),
                    target: Box::new(args_raw[1].clone()),
                    max_depth,
                })
            }
            "somepath" => {
                if args_raw.len() < 2 {
                    return Err("somepath() requires 2 arguments".to_string());
                }
                Ok(QueryExpr::SomePath {
                    from: Box::new(args_raw[0].clone()),
                    to: Box::new(args_raw[1].clone()),
                })
            }
            "allpaths" => {
                if args_raw.len() < 2 {
                    return Err("allpaths() requires 2 arguments".to_string());
                }
                Ok(QueryExpr::AllPaths {
                    from: Box::new(args_raw[0].clone()),
                    to: Box::new(args_raw[1].clone()),
                })
            }
            "kind" => {
                if args_raw.len() < 2 {
                    return Err("kind() requires 2 arguments".to_string());
                }
                let pat = expr_as_string(&args_raw[0]);
                Ok(QueryExpr::Kind {
                    pattern: pat,
                    input: Box::new(args_raw[1].clone()),
                })
            }
            "filter" => {
                if args_raw.len() < 2 {
                    return Err("filter() requires 2 arguments".to_string());
                }
                let pat = expr_as_string(&args_raw[0]);
                Ok(QueryExpr::Filter {
                    pattern: pat,
                    input: Box::new(args_raw[1].clone()),
                })
            }
            "attr" => {
                if args_raw.len() < 3 {
                    return Err("attr() requires 3 arguments".to_string());
                }
                Ok(QueryExpr::Attr {
                    attr_name: expr_as_string(&args_raw[0]),
                    pattern: expr_as_string(&args_raw[1]),
                    input: Box::new(args_raw[2].clone()),
                })
            }
            "tests" => {
                let inner = args_raw
                    .into_iter()
                    .next()
                    .ok_or_else(|| "tests() requires 1 argument".to_string())?;
                Ok(QueryExpr::Tests(Box::new(inner)))
            }
            "siblings" => {
                let inner = args_raw
                    .into_iter()
                    .next()
                    .ok_or_else(|| "siblings() requires 1 argument".to_string())?;
                Ok(QueryExpr::Siblings(Box::new(inner)))
            }
            "buildfiles" | "loadfiles" => {
                let inner = args_raw
                    .into_iter()
                    .next()
                    .ok_or_else(|| "buildfiles() requires 1 argument".to_string())?;
                Ok(QueryExpr::BuildFiles(Box::new(inner)))
            }
            other => Err(format!("Unknown query function: '{other}'")),
        }
    } else {
        Ok(QueryExpr::TargetPattern(tok))
    }
}

fn expr_as_string(expr: &QueryExpr) -> String {
    match expr {
        QueryExpr::TargetPattern(s) => s.clone(),
        _ => String::new(),
    }
}

fn expr_as_usize(expr: &QueryExpr) -> Option<usize> {
    match expr {
        QueryExpr::TargetPattern(s) => s.parse().ok(),
        _ => None,
    }
}

/// Evaluates `QueryExpr` over a map of loaded `Package`s.
pub struct QueryEnvironment {
    pub packages: BTreeMap<PackageIdentifier, Package>,
}

impl QueryEnvironment {
    pub fn new(packages: BTreeMap<PackageIdentifier, Package>) -> Self {
        Self { packages }
    }

    pub fn evaluate(&self, expr: &QueryExpr) -> Result<Vec<Target>, String> {
        let mut vars = HashMap::new();
        let labels = self.eval_labels(expr, &mut vars)?;
        let mut out = Vec::new();
        for lbl in labels {
            if let Some(t) = self.lookup_target(&lbl) {
                out.push(t);
            }
        }
        Ok(out)
    }

    fn eval_labels(
        &self,
        expr: &QueryExpr,
        vars: &mut HashMap<String, BTreeSet<Label>>,
    ) -> Result<BTreeSet<Label>, String> {
        match expr {
            QueryExpr::TargetPattern(pat_str) => self.resolve_pattern(pat_str),
            QueryExpr::Var(name) => vars
                .get(name)
                .cloned()
                .ok_or_else(|| format!("Undefined query variable '${name}'")),
            QueryExpr::Let {
                var_name,
                val,
                body,
            } => {
                let bound = self.eval_labels(val, vars)?;
                vars.insert(var_name.clone(), bound);
                self.eval_labels(body, vars)
            }
            QueryExpr::Union(a, b) => {
                let mut s = self.eval_labels(a, vars)?;
                s.extend(self.eval_labels(b, vars)?);
                Ok(s)
            }
            QueryExpr::Except(a, b) => {
                let s1 = self.eval_labels(a, vars)?;
                let s2 = self.eval_labels(b, vars)?;
                Ok(s1.difference(&s2).cloned().collect())
            }
            QueryExpr::Intersect(a, b) => {
                let s1 = self.eval_labels(a, vars)?;
                let s2 = self.eval_labels(b, vars)?;
                Ok(s1.intersection(&s2).cloned().collect())
            }
            QueryExpr::Kind { pattern, input } => {
                let base = self.eval_labels(input, vars)?;
                let pat_clean = pattern.trim_end_matches(" rule");
                Ok(base
                    .into_iter()
                    .filter(|lbl| {
                        self.lookup_target(lbl)
                            .map(|t| t.target_kind().contains(pat_clean))
                            .unwrap_or(false)
                    })
                    .collect())
            }
            QueryExpr::Filter { pattern, input } => {
                let base = self.eval_labels(input, vars)?;
                Ok(base
                    .into_iter()
                    .filter(|lbl| lbl.canonical_form().contains(pattern))
                    .collect())
            }
            QueryExpr::Attr {
                attr_name,
                pattern,
                input,
            } => {
                let base = self.eval_labels(input, vars)?;
                Ok(base
                    .into_iter()
                    .filter(|lbl| {
                        if let Some(Target::Rule(r)) = self.lookup_target(lbl) {
                            if let Some(val) = r.attrs.get(attr_name) {
                                return val.to_string().contains(pattern);
                            }
                        }
                        false
                    })
                    .collect())
            }
            QueryExpr::Tests(inner) => {
                let base = self.eval_labels(inner, vars)?;
                Ok(base
                    .into_iter()
                    .filter(|lbl| {
                        if let Some(Target::Rule(r)) = self.lookup_target(lbl) {
                            r.is_test_rule()
                        } else {
                            false
                        }
                    })
                    .collect())
            }
            QueryExpr::Siblings(inner) => {
                let base = self.eval_labels(inner, vars)?;
                let mut pkgs = BTreeSet::new();
                for lbl in base {
                    pkgs.insert(lbl.package_identifier().clone());
                }
                let mut out = BTreeSet::new();
                for pkg_id in pkgs {
                    if let Some(pkg) = self.packages.get(&pkg_id) {
                        for t in pkg.targets.values() {
                            out.insert(t.label().clone());
                        }
                    }
                }
                Ok(out)
            }
            QueryExpr::BuildFiles(inner) => {
                let base = self.eval_labels(inner, vars)?;
                let mut out = BTreeSet::new();
                for lbl in base {
                    if let Ok(build_lbl) = Label::create(lbl.package_identifier().clone(), "BUILD")
                    {
                        out.insert(build_lbl);
                    }
                }
                Ok(out)
            }
            QueryExpr::Deps { target, max_depth } => {
                let roots = self.eval_labels(target, vars)?;
                Ok(self.bfs_deps(&roots, *max_depth))
            }
            QueryExpr::Rdeps {
                universe,
                target,
                max_depth,
            } => {
                let univ = self.eval_labels(universe, vars)?;
                let targets = self.eval_labels(target, vars)?;
                let univ_closed = self.bfs_deps(&univ, None);
                Ok(self.bfs_rdeps(&univ_closed, &targets, *max_depth))
            }
            QueryExpr::SomePath { from, to } | QueryExpr::AllPaths { from, to } => {
                let starts = self.eval_labels(from, vars)?;
                let ends = self.eval_labels(to, vars)?;
                let forward = self.bfs_deps(&starts, None);
                let backward = self.bfs_rdeps(&forward, &ends, None);
                Ok(forward.intersection(&backward).cloned().collect())
            }
        }
    }

    fn bfs_deps(&self, roots: &BTreeSet<Label>, max_depth: Option<usize>) -> BTreeSet<Label> {
        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::new();
        for r in roots {
            visited.insert(r.clone());
            queue.push_back((r.clone(), 0usize));
        }

        while let Some((curr, depth)) = queue.pop_front() {
            if let Some(max_d) = max_depth {
                if depth >= max_d {
                    continue;
                }
            }
            for dep in self.direct_deps(&curr) {
                if visited.insert(dep.clone()) {
                    queue.push_back((dep, depth + 1));
                }
            }
        }
        visited
    }

    fn bfs_rdeps(
        &self,
        universe: &BTreeSet<Label>,
        targets: &BTreeSet<Label>,
        max_depth: Option<usize>,
    ) -> BTreeSet<Label> {
        let mut rdeps_map: HashMap<Label, Vec<Label>> = HashMap::new();
        for u in universe {
            for dep in self.direct_deps(u) {
                rdeps_map.entry(dep).or_default().push(u.clone());
            }
        }

        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::new();
        for t in targets {
            if universe.contains(t) {
                visited.insert(t.clone());
                queue.push_back((t.clone(), 0usize));
            }
        }

        while let Some((curr, depth)) = queue.pop_front() {
            if let Some(max_d) = max_depth {
                if depth >= max_d {
                    continue;
                }
            }
            if let Some(parents) = rdeps_map.get(&curr) {
                for p in parents {
                    if visited.insert(p.clone()) {
                        queue.push_back((p.clone(), depth + 1));
                    }
                }
            }
        }
        visited
    }

    pub fn direct_deps(&self, label: &Label) -> Vec<Label> {
        let mut out = Vec::new();
        if let Some(Target::Rule(r)) = self.lookup_target(label) {
            for raw_dep in r.deps.iter().chain(r.srcs.iter()).chain(r.tools.iter()) {
                if let Some(dep_lbl) = Self::resolve_relative_label(label.package_identifier(), raw_dep)
                {
                    out.push(dep_lbl);
                }
            }
        }
        out
    }

    pub fn resolve_relative_label(pkg_id: &PackageIdentifier, raw: &str) -> Option<Label> {
        if raw.starts_with("//") || raw.starts_with('@') {
            Label::parse_canonical(raw).ok()
        } else {
            let name = raw.strip_prefix(':').unwrap_or(raw);
            Label::create(pkg_id.clone(), name).ok()
        }
    }

    fn lookup_target(&self, label: &Label) -> Option<Target> {
        let pkg = self.packages.get(label.package_identifier())?;
        pkg.targets.get(label.name()).cloned()
    }

    fn resolve_pattern(&self, pat_str: &str) -> Result<BTreeSet<Label>, String> {
        let pat = TargetPattern::parse(pat_str, &PathFragment::empty())
            .map_err(|e| e.to_string())?;
        let mut out = BTreeSet::new();
        match pat {
            TargetPattern::SingleTarget(lbl) => {
                out.insert(lbl);
            }
            TargetPattern::TargetsInPackage {
                package,
                include_files,
            } => {
                if let Some(pkg) = self.packages.get(&package) {
                    for t in pkg.targets.values() {
                        if include_files || matches!(t, Target::Rule(_)) {
                            out.insert(t.label().clone());
                        }
                    }
                }
            }
            TargetPattern::TargetsBelowDirectory {
                directory,
                include_files,
            } => {
                for (pkg_id, pkg) in &self.packages {
                    if pkg_id
                        .package_fragment()
                        .starts_with(directory.package_fragment())
                    {
                        for t in pkg.targets.values() {
                            if include_files || matches!(t, Target::Rule(_)) {
                                out.insert(t.label().clone());
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Formats query results according to `--output={label,label_kind,build,package,graph,xml}`.
    pub fn format_results(&self, targets: &[Target], output_format: &str) -> String {
        let mut out = String::new();
        match output_format {
            "label_kind" => {
                for t in targets {
                    out.push_str(&format!(
                        "{} {}\n",
                        t.target_kind(),
                        t.label().canonical_form()
                    ));
                }
            }
            "package" => {
                let mut pkgs = BTreeSet::new();
                for t in targets {
                    pkgs.insert(t.label().package_name().to_string());
                }
                for p in pkgs {
                    out.push_str(&p);
                    out.push('\n');
                }
            }
            "build" => {
                for t in targets {
                    if let Target::Rule(r) = t {
                        out.push_str(&format!(
                            "{}(\n  name = \"{}\",\n)\n",
                            r.rule_class,
                            r.label.name()
                        ));
                    }
                }
            }
            "graph" => {
                out.push_str("digraph mygraph {\n  node [shape=box];\n");
                for t in targets {
                    let lbl = t.label();
                    out.push_str(&format!("  \"{}\"\n", lbl.canonical_form()));
                    for dep in self.direct_deps(lbl) {
                        out.push_str(&format!(
                            "  \"{}\" -> \"{}\"\n",
                            lbl.canonical_form(),
                            dep.canonical_form()
                        ));
                    }
                }
                out.push_str("}\n");
            }
            "xml" => {
                out.push_str("<?xml version=\"1.1\" encoding=\"UTF-8\" standalone=\"no\"?>\n<query version=\"2\">\n");
                for t in targets {
                    match t {
                        Target::Rule(r) => {
                            out.push_str(&format!(
                                "  <rule class=\"{}\" name=\"{}\"/>\n",
                                r.rule_class,
                                r.label.canonical_form()
                            ));
                        }
                        other => {
                            out.push_str(&format!(
                                "  <source-file name=\"{}\"/>\n",
                                other.label().canonical_form()
                            ));
                        }
                    }
                }
                out.push_str("</query>\n");
            }
            _ => {
                for t in targets {
                    out.push_str(&t.label().canonical_form());
                    out.push('\n');
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bazel_vfs::{FileSystem, InMemoryFileSystem};
    use std::sync::Arc;

    #[test]
    fn test_query_deps_rdeps_kind_and_set_ops() {
        let fs = Arc::new(InMemoryFileSystem::default());
        let ws = PathFragment::create("/workspace");
        let pkg_dir = ws.get_relative_str("app");
        fs.create_directory_and_parents(&pkg_dir).unwrap();
        let build_src = r#"
cc_library(
    name = "util",
    srcs = ["util.cc"],
)
cc_library(
    name = "core",
    srcs = ["core.cc"],
    deps = [":util"],
)
cc_binary(
    name = "server",
    srcs = ["server.cc"],
    deps = [":core"],
)
cc_test(
    name = "core_test",
    srcs = ["core_test.cc"],
    deps = [":core"],
)
"#;
        let pkg_id = PackageIdentifier::create_in_main_repo("app").unwrap();
        let pkg = Package::evaluate_build_file(
            pkg_id.clone(),
            &ws,
            pkg_dir.get_relative_str("BUILD"),
            build_src,
            fs,
        )
        .unwrap();

        let mut packages = BTreeMap::new();
        packages.insert(pkg_id, pkg);
        let qenv = QueryEnvironment::new(packages);

        // 1. deps(//app:server)
        let expr1 = QueryExpr::parse("kind('cc_library', deps(//app:server))").unwrap();
        let res1 = qenv.evaluate(&expr1).unwrap();
        let labels1: Vec<String> = res1.iter().map(|t| t.label().canonical_form()).collect();
        assert_eq!(labels1, vec!["//app:core", "//app:util"]);

        // 2. rdeps(//app:all, //app:util)
        let expr2 = QueryExpr::parse("rdeps(//app:all, //app:util) - //app:util").unwrap();
        let res2 = qenv.evaluate(&expr2).unwrap();
        let labels2: Vec<String> = res2.iter().map(|t| t.label().canonical_form()).collect();
        assert_eq!(
            labels2,
            vec!["//app:core", "//app:core_test", "//app:server"]
        );

        // 3. tests(//app:all)
        let expr3 = QueryExpr::parse("tests(//app:all)").unwrap();
        let res3 = qenv.evaluate(&expr3).unwrap();
        assert_eq!(res3.len(), 1);
        assert_eq!(res3[0].label().canonical_form(), "//app:core_test");
    }
}
