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

//! Starlark Interpreter & Bazel Dialect (`Depset`, `SelectorList`, `Provider`, `Struct`,
//! `BUILD` / `.bzl` / `MODULE.bazel` evaluation).
//! Mirrors `src/main/java/net/starlark/java` and `build/lib/packages`.

use bazel_cmdline::Label;
use bazel_vfs::{evaluate_glob, FileSystem, PathFragment};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StarlarkError(pub String);

impl fmt::Display for StarlarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StarlarkError {}

/// Traversal order for `depset` (`NestedSet` in Java).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DepsetOrder {
    #[default]
    Default,
    Postorder,
    Preorder,
    Topological,
}

impl DepsetOrder {
    pub fn parse(s: &str) -> Result<Self, StarlarkError> {
        match s {
            "default" => Ok(Self::Default),
            "postorder" => Ok(Self::Postorder),
            "preorder" => Ok(Self::Preorder),
            "topological" => Ok(Self::Topological),
            other => Err(StarlarkError(format!("Invalid depset order: '{other}'"))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Postorder => "postorder",
            Self::Preorder => "preorder",
            Self::Topological => "topological",
        }
    }
}

/// Bazel `depset` value supporting O(1) transitive nesting and ordered deduplicated flattening.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Depset {
    pub order: DepsetOrder,
    pub direct: Vec<Value>,
    pub transitive: Vec<Arc<Depset>>,
}

impl Depset {
    pub fn new(
        order: DepsetOrder,
        direct: Vec<Value>,
        transitive: Vec<Arc<Depset>>,
    ) -> Result<Self, StarlarkError> {
        for t in &transitive {
            if t.order != DepsetOrder::Default
                && order != DepsetOrder::Default
                && t.order != order
            {
                return Err(StarlarkError(format!(
                    "Order '{}' is incompatible with transitive depset order '{}'",
                    order.as_str(),
                    t.order.as_str()
                )));
            }
        }
        Ok(Self {
            order,
            direct,
            transitive,
        })
    }

    pub fn to_list(&self) -> Vec<Value> {
        let mut raw = Vec::new();
        match self.order {
            DepsetOrder::Preorder => {
                self.collect_preorder(&mut raw);
            }
            DepsetOrder::Default | DepsetOrder::Postorder | DepsetOrder::Topological => {
                self.collect_postorder(&mut raw);
            }
        }
        let mut deduped = Vec::new();
        for item in raw {
            if !deduped.contains(&item) {
                deduped.push(item);
            }
        }
        deduped
    }

    fn collect_postorder(&self, out: &mut Vec<Value>) {
        for t in &self.transitive {
            t.collect_postorder(out);
        }
        for d in &self.direct {
            out.push(d.clone());
        }
    }

    fn collect_preorder(&self, out: &mut Vec<Value>) {
        for d in &self.direct {
            out.push(d.clone());
        }
        for t in &self.transitive {
            t.collect_preorder(out);
        }
    }
}

/// Element of a `select(...)` concatenation expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectorElement {
    Select(BTreeMap<String, Value>),
    Native(Value),
}

/// Represents a configurable attribute value (`select({...}) + [...]`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SelectorList {
    pub elements: Vec<SelectorElement>,
}

impl SelectorList {
    pub fn from_select(map: BTreeMap<String, Value>) -> Self {
        Self {
            elements: vec![SelectorElement::Select(map)],
        }
    }

    /// Resolves the `SelectorList` against a set of active `config_setting` conditions
    /// (falling back to `//conditions:default`).
    pub fn resolve(&self, active_conditions: &[String]) -> Result<Value, StarlarkError> {
        let mut acc: Option<Value> = None;
        for elem in &self.elements {
            let chosen = match elem {
                SelectorElement::Native(v) => v.clone(),
                SelectorElement::Select(branches) => {
                    let mut matched = None;
                    for cond in active_conditions {
                        if let Some(v) = branches.get(cond) {
                            matched = Some(v.clone());
                            break;
                        }
                    }
                    if matched.is_none() {
                        matched = branches.get("//conditions:default").cloned();
                    }
                    if matched.is_none() {
                        matched = branches.values().next().cloned();
                    }
                    matched.unwrap_or(Value::None)
                }
            };
            acc = Some(match acc {
                None => chosen,
                Some(prev) => prev.add(&chosen)?,
            });
        }
        Ok(acc.unwrap_or(Value::None))
    }
}

/// Runtime value in the Starlark interpreter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    String(String),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(BTreeMap<String, Value>),
    Depset(Arc<Depset>),
    SelectorList(SelectorList),
    Label(Label),
    Struct(BTreeMap<String, Value>),
    Provider {
        name: String,
    },
    ProviderInstance {
        provider_name: String,
        fields: BTreeMap<String, Value>,
    },
    RuleCallable {
        rule_class: String,
        executable: bool,
        test: bool,
    },
    UserFunction(Arc<UserFunctionDef>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamDef {
    pub name: String,
    pub default: Option<Expr>,
    pub is_args: bool,
    pub is_kwargs: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserFunctionDef {
    pub name: String,
    pub params: Vec<ParamDef>,
    pub body: Vec<Stmt>,
}

impl Value {
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::None => false,
            Self::Bool(b) => *b,
            Self::Int(i) => *i != 0,
            Self::String(s) => !s.is_empty(),
            Self::List(l) | Self::Tuple(l) => !l.is_empty(),
            Self::Dict(d) | Self::Struct(d) => !d.is_empty(),
            Self::Depset(d) => !d.to_list().is_empty(),
            Self::SelectorList(s) => !s.elements.is_empty(),
            _ => true,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::None => "NoneType",
            Self::Bool(_) => "bool",
            Self::Int(_) => "int",
            Self::String(_) => "string",
            Self::List(_) => "list",
            Self::Tuple(_) => "tuple",
            Self::Dict(_) => "dict",
            Self::Depset(_) => "depset",
            Self::SelectorList(_) => "select",
            Self::Label(_) => "Label",
            Self::Struct(_) => "struct",
            Self::Provider { .. } => "Provider",
            Self::ProviderInstance { .. } => "ProviderInstance",
            Self::RuleCallable { .. } => "rule",
            Self::UserFunction(_) => "function",
        }
    }

    pub fn add(&self, other: &Value) -> Result<Value, StarlarkError> {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => Ok(Self::Int(a.wrapping_add(*b))),
            (Self::String(a), Self::String(b)) => Ok(Self::String(format!("{a}{b}"))),
            (Self::List(a), Self::List(b)) => {
                let mut v = a.clone();
                v.extend(b.iter().cloned());
                Ok(Self::List(v))
            }
            (Self::Tuple(a), Self::Tuple(b)) => {
                let mut v = a.clone();
                v.extend(b.iter().cloned());
                Ok(Self::Tuple(v))
            }
            (Self::SelectorList(s1), Self::SelectorList(s2)) => {
                let mut elements = s1.elements.clone();
                elements.extend(s2.elements.iter().cloned());
                Ok(Self::SelectorList(SelectorList { elements }))
            }
            (Self::SelectorList(s1), rhs) => {
                let mut elements = s1.elements.clone();
                elements.push(SelectorElement::Native(rhs.clone()));
                Ok(Self::SelectorList(SelectorList { elements }))
            }
            (lhs, Self::SelectorList(s2)) => {
                let mut elements = vec![SelectorElement::Native(lhs.clone())];
                elements.extend(s2.elements.iter().cloned());
                Ok(Self::SelectorList(SelectorList { elements }))
            }
            (Self::Dict(d1), Self::Dict(d2)) => {
                let mut m = d1.clone();
                for (k, v) in d2 {
                    m.insert(k.clone(), v.clone());
                }
                Ok(Self::Dict(m))
            }
            _ => Err(StarlarkError(format!(
                "unsupported binary operation: {} + {}",
                self.type_name(),
                other.type_name()
            ))),
        }
    }

    pub fn to_string_list(&self) -> Vec<String> {
        match self {
            Self::List(items) | Self::Tuple(items) => items
                .iter()
                .filter_map(|v| match v {
                    Self::String(s) => Some(s.clone()),
                    Self::Label(l) => Some(l.canonical_form()),
                    _ => None,
                })
                .collect(),
            Self::SelectorList(sel) => sel
                .resolve(&[])
                .map(|v| v.to_string_list())
                .unwrap_or_default(),
            Self::Depset(d) => Value::List(d.to_list()).to_string_list(),
            _ => Vec::new(),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("None"),
            Self::Bool(true) => f.write_str("True"),
            Self::Bool(false) => f.write_str("False"),
            Self::Int(i) => write!(f, "{i}"),
            Self::String(s) => f.write_str(s),
            Self::Label(l) => write!(f, "{l}"),
            Self::List(items) => {
                f.write_str("[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item:?}")?;
                }
                f.write_str("]")
            }
            Self::Tuple(items) => {
                f.write_str("(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item:?}")?;
                }
                f.write_str(")")
            }
            Self::Dict(map) | Self::Struct(map) => {
                f.write_str("{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{k:?}: {v}")?;
                }
                f.write_str("}")
            }
            Self::Depset(d) => write!(f, "depset({:?})", d.to_list()),
            Self::SelectorList(s) => write!(f, "select({:?})", s.elements),
            Self::Provider { name } => write!(f, "<provider {name}>"),
            Self::ProviderInstance {
                provider_name,
                fields,
            } => write!(f, "{provider_name}({fields:?})"),
            Self::RuleCallable { rule_class, .. } => write!(f, "<rule {rule_class}>"),
            Self::UserFunction(func) => write!(f, "<function {}>", func.name),
        }
    }
}

/// Recorded rule call during `BUILD` file evaluation (e.g. `cc_binary(name = "foo", ...)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedRuleCall {
    pub rule_class: String,
    pub kwargs: BTreeMap<String, Value>,
}

/// Recorded `load("...", "sym1", alias = "sym2")` statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadStatement {
    pub module_label: String,
    pub symbols: Vec<(String, String)>, // (local_name, exported_name)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    NotIn,
    And,
    Or,
    BitOr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Literal(Value),
    Ident(String),
    List(Vec<Expr>),
    Tuple(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
    ListComp {
        body: Box<Expr>,
        var_names: Vec<String>,
        iter: Box<Expr>,
        cond: Option<Box<Expr>>,
    },
    DictComp {
        key: Box<Expr>,
        val: Box<Expr>,
        var_names: Vec<String>,
        iter: Box<Expr>,
        cond: Option<Box<Expr>>,
    },
    UnaryNot(Box<Expr>),
    UnaryNeg(Box<Expr>),
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    IfExpr {
        body: Box<Expr>,
        cond: Box<Expr>,
        else_body: Box<Expr>,
    },
    Call {
        callee: Box<Expr>,
        pos_args: Vec<Expr>,
        kw_args: Vec<(String, Expr)>,
    },
    Attr {
        target: Box<Expr>,
        attr: String,
    },
    Index {
        target: Box<Expr>,
        index: Box<Expr>,
    },
    Slice {
        target: Box<Expr>,
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Load(LoadStatement),
    Assign {
        target: String,
        op_add: bool,
        expr: Expr,
    },
    TupleAssign {
        targets: Vec<String>,
        expr: Expr,
    },
    Expr(Expr),
    Def(Arc<UserFunctionDef>),
    If {
        branches: Vec<(Expr, Vec<Stmt>)>,
        else_body: Vec<Stmt>,
    },
    For {
        var_names: Vec<String>,
        iter: Expr,
        body: Vec<Stmt>,
    },
    Return(Option<Expr>),
    Pass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Ident(String),
    Int(i64),
    Str(String),
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Dot,
    Assign,
    PlusAssign,
    Plus,
    Minus,
    Star,
    StarStar,
    Slash,
    SlashSlash,
    Percent,
    Pipe,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    Newline,
    Indent,
    Dedent,
    Eof,
}

fn lex(source: &str) -> Result<Vec<Token>, StarlarkError> {
    let mut tokens = Vec::new();
    let mut indent_stack: Vec<usize> = vec![0];
    let mut paren_depth: usize = 0;

    for raw_line in source.lines() {
        // Count leading spaces if not inside parens/brackets/braces
        let bytes = raw_line.as_bytes();
        let mut idx = 0;
        let mut indent = 0;
        while idx < bytes.len() && (bytes[idx] == b' ' || bytes[idx] == b'\t') {
            indent += if bytes[idx] == b'\t' { 4 } else { 1 };
            idx += 1;
        }
        // Skip blank or comment-only lines
        if idx >= bytes.len() || bytes[idx] == b'#' {
            continue;
        }

        if paren_depth == 0 {
            let current_indent = *indent_stack.last().unwrap_or(&0);
            if indent > current_indent {
                indent_stack.push(indent);
                tokens.push(Token::Indent);
            } else {
                while indent < *indent_stack.last().unwrap_or(&0) {
                    indent_stack.pop();
                    tokens.push(Token::Dedent);
                }
            }
        }

        while idx < bytes.len() {
            let b = bytes[idx];
            if b == b' ' || b == b'\t' || b == b'\r' {
                idx += 1;
                continue;
            }
            if b == b'#' {
                break;
            }
            if b == b'\\' && idx + 1 == bytes.len() {
                // Line continuation
                break;
            }

            // Raw string prefix r"..." or r'...'
            if (b == b'r' || b == b'R')
                && idx + 1 < bytes.len()
                && (bytes[idx + 1] == b'"' || bytes[idx + 1] == b'\'')
            {
                idx += 1;
                let (s, next_idx) = lex_string(raw_line, idx, true)?;
                tokens.push(Token::Str(s));
                idx = next_idx;
                continue;
            }

            if b == b'"' || b == b'\'' {
                let (s, next_idx) = lex_string(raw_line, idx, false)?;
                tokens.push(Token::Str(s));
                idx = next_idx;
                continue;
            }

            if b.is_ascii_digit() {
                let start = idx;
                while idx < bytes.len() && bytes[idx].is_ascii_digit() {
                    idx += 1;
                }
                let val = raw_line[start..idx]
                    .parse::<i64>()
                    .map_err(|e| StarlarkError(format!("invalid integer literal: {e}")))?;
                tokens.push(Token::Int(val));
                continue;
            }

            if b.is_ascii_alphabetic() || b == b'_' {
                let start = idx;
                while idx < bytes.len()
                    && (bytes[idx].is_ascii_alphanumeric() || bytes[idx] == b'_')
                {
                    idx += 1;
                }
                tokens.push(Token::Ident(raw_line[start..idx].to_string()));
                continue;
            }

            match b {
                b'(' => {
                    paren_depth += 1;
                    tokens.push(Token::LParen);
                    idx += 1;
                }
                b')' => {
                    paren_depth = paren_depth.saturating_sub(1);
                    tokens.push(Token::RParen);
                    idx += 1;
                }
                b'[' => {
                    paren_depth += 1;
                    tokens.push(Token::LBracket);
                    idx += 1;
                }
                b']' => {
                    paren_depth = paren_depth.saturating_sub(1);
                    tokens.push(Token::RBracket);
                    idx += 1;
                }
                b'{' => {
                    paren_depth += 1;
                    tokens.push(Token::LBrace);
                    idx += 1;
                }
                b'}' => {
                    paren_depth = paren_depth.saturating_sub(1);
                    tokens.push(Token::RBrace);
                    idx += 1;
                }
                b',' => {
                    tokens.push(Token::Comma);
                    idx += 1;
                }
                b':' => {
                    tokens.push(Token::Colon);
                    idx += 1;
                }
                b'.' => {
                    tokens.push(Token::Dot);
                    idx += 1;
                }
                b'+' => {
                    if idx + 1 < bytes.len() && bytes[idx + 1] == b'=' {
                        tokens.push(Token::PlusAssign);
                        idx += 2;
                    } else {
                        tokens.push(Token::Plus);
                        idx += 1;
                    }
                }
                b'-' => {
                    tokens.push(Token::Minus);
                    idx += 1;
                }
                b'*' => {
                    if idx + 1 < bytes.len() && bytes[idx + 1] == b'*' {
                        tokens.push(Token::StarStar);
                        idx += 2;
                    } else {
                        tokens.push(Token::Star);
                        idx += 1;
                    }
                }
                b'/' => {
                    if idx + 1 < bytes.len() && bytes[idx + 1] == b'/' {
                        tokens.push(Token::SlashSlash);
                        idx += 2;
                    } else {
                        tokens.push(Token::Slash);
                        idx += 1;
                    }
                }
                b'%' => {
                    tokens.push(Token::Percent);
                    idx += 1;
                }
                b'|' => {
                    tokens.push(Token::Pipe);
                    idx += 1;
                }
                b'=' => {
                    if idx + 1 < bytes.len() && bytes[idx + 1] == b'=' {
                        tokens.push(Token::EqEq);
                        idx += 2;
                    } else {
                        tokens.push(Token::Assign);
                        idx += 1;
                    }
                }
                b'!' => {
                    if idx + 1 < bytes.len() && bytes[idx + 1] == b'=' {
                        tokens.push(Token::NotEq);
                        idx += 2;
                    } else {
                        idx += 1;
                    }
                }
                b'<' => {
                    if idx + 1 < bytes.len() && bytes[idx + 1] == b'=' {
                        tokens.push(Token::Le);
                        idx += 2;
                    } else {
                        tokens.push(Token::Lt);
                        idx += 1;
                    }
                }
                b'>' => {
                    if idx + 1 < bytes.len() && bytes[idx + 1] == b'=' {
                        tokens.push(Token::Ge);
                        idx += 2;
                    } else {
                        tokens.push(Token::Gt);
                        idx += 1;
                    }
                }
                _ => {
                    idx += 1;
                }
            }
        }

        if paren_depth == 0 {
            tokens.push(Token::Newline);
        }
    }

    while indent_stack.len() > 1 {
        indent_stack.pop();
        tokens.push(Token::Dedent);
    }
    tokens.push(Token::Eof);
    Ok(tokens)
}

fn lex_string(line: &str, start: usize, raw: bool) -> Result<(String, usize), StarlarkError> {
    let bytes = line.as_bytes();
    let quote = bytes[start];
    let is_triple = start + 2 < bytes.len()
        && bytes[start + 1] == quote
        && bytes[start + 2] == quote;
    let mut idx = if is_triple { start + 3 } else { start + 1 };
    let mut out = String::new();

    while idx < bytes.len() {
        if is_triple {
            if idx + 2 < bytes.len()
                && bytes[idx] == quote
                && bytes[idx + 1] == quote
                && bytes[idx + 2] == quote
            {
                return Ok((out, idx + 3));
            }
        } else if bytes[idx] == quote {
            return Ok((out, idx + 1));
        }

        if !raw && bytes[idx] == b'\\' && idx + 1 < bytes.len() {
            idx += 1;
            match bytes[idx] {
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'\\' => out.push('\\'),
                b'"' => out.push('"'),
                b'\'' => out.push('\''),
                other => {
                    out.push('\\');
                    out.push(other as char);
                }
            }
            idx += 1;
        } else {
            let ch = line[idx..].chars().next().unwrap();
            out.push(ch);
            idx += ch.len_utf8();
        }
    }
    Ok((out, idx))
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn next(&mut self) -> Token {
        let tok = self.peek().clone();
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn eat(&mut self, expected: &Token) -> Result<(), StarlarkError> {
        let got = self.next();
        if &got == expected {
            Ok(())
        } else {
            Err(StarlarkError(format!(
                "Expected {expected:?}, got {got:?}"
            )))
        }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Token::Newline) {
            self.next();
        }
    }

    fn parse_module(&mut self) -> Result<Vec<Stmt>, StarlarkError> {
        let mut stmts = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek(), Token::Eof | Token::Dedent) {
            stmts.push(self.parse_stmt()?);
            self.skip_newlines();
        }
        Ok(stmts)
    }

    fn parse_suite(&mut self) -> Result<Vec<Stmt>, StarlarkError> {
        self.eat(&Token::Colon)?;
        if matches!(self.peek(), Token::Newline) {
            self.next();
            self.eat(&Token::Indent)?;
            let mut stmts = Vec::new();
            self.skip_newlines();
            while !matches!(self.peek(), Token::Dedent | Token::Eof) {
                stmts.push(self.parse_stmt()?);
                self.skip_newlines();
            }
            if matches!(self.peek(), Token::Dedent) {
                self.next();
            }
            Ok(stmts)
        } else {
            let s = self.parse_stmt()?;
            Ok(vec![s])
        }
    }

    fn parse_stmt(&mut self) -> Result<Stmt, StarlarkError> {
        match self.peek().clone() {
            Token::Ident(ref kw) if kw == "def" => {
                self.next();
                let Token::Ident(name) = self.next() else {
                    return Err(StarlarkError("Expected function name after def".to_string()));
                };
                self.eat(&Token::LParen)?;
                let mut params = Vec::new();
                while !matches!(self.peek(), Token::RParen | Token::Eof) {
                    let mut is_args = false;
                    let mut is_kwargs = false;
                    if matches!(self.peek(), Token::Star) {
                        self.next();
                        if matches!(self.peek(), Token::Comma) {
                            self.next();
                            continue;
                        }
                        is_args = true;
                    } else if matches!(self.peek(), Token::StarStar) {
                        self.next();
                        is_kwargs = true;
                    }
                    let Token::Ident(pname) = self.next() else {
                        return Err(StarlarkError("Expected parameter name".to_string()));
                    };
                    let default = if matches!(self.peek(), Token::Assign) {
                        self.next();
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    params.push(ParamDef {
                        name: pname,
                        default,
                        is_args,
                        is_kwargs,
                    });
                    if matches!(self.peek(), Token::Comma) {
                        self.next();
                    }
                }
                self.eat(&Token::RParen)?;
                let body = self.parse_suite()?;
                Ok(Stmt::Def(Arc::new(UserFunctionDef { name, params, body })))
            }
            Token::Ident(ref kw) if kw == "if" => {
                self.next();
                let cond = self.parse_expr()?;
                let body = self.parse_suite()?;
                let mut branches = vec![(cond, body)];
                let mut else_body = Vec::new();
                self.skip_newlines();
                while let Token::Ident(ref next_kw) = self.peek().clone() {
                    if next_kw == "elif" {
                        self.next();
                        let econd = self.parse_expr()?;
                        let ebody = self.parse_suite()?;
                        branches.push((econd, ebody));
                        self.skip_newlines();
                    } else if next_kw == "else" {
                        self.next();
                        else_body = self.parse_suite()?;
                        break;
                    } else {
                        break;
                    }
                }
                Ok(Stmt::If {
                    branches,
                    else_body,
                })
            }
            Token::Ident(ref kw) if kw == "for" => {
                self.next();
                let var_names = self.parse_var_list()?;
                let Token::Ident(in_kw) = self.next() else {
                    return Err(StarlarkError("Expected 'in' in for loop".to_string()));
                };
                if in_kw != "in" {
                    return Err(StarlarkError("Expected 'in' in for loop".to_string()));
                }
                let iter = self.parse_expr()?;
                let body = self.parse_suite()?;
                Ok(Stmt::For {
                    var_names,
                    iter,
                    body,
                })
            }
            Token::Ident(ref kw) if kw == "return" => {
                self.next();
                if matches!(self.peek(), Token::Newline | Token::Dedent | Token::Eof) {
                    Ok(Stmt::Return(None))
                } else {
                    Ok(Stmt::Return(Some(self.parse_expr()?)))
                }
            }
            Token::Ident(ref kw) if kw == "pass" => {
                self.next();
                Ok(Stmt::Pass)
            }
            Token::Ident(ref kw) if kw == "load" => {
                self.next();
                self.eat(&Token::LParen)?;
                let Token::Str(module_label) = self.next() else {
                    return Err(StarlarkError(
                        "First argument to load() must be a string literal".to_string(),
                    ));
                };
                if matches!(self.peek(), Token::Comma) {
                    self.next();
                }
                let mut symbols = Vec::new();
                while !matches!(self.peek(), Token::RParen | Token::Eof) {
                    match self.next() {
                        Token::Str(sym) => {
                            symbols.push((sym.clone(), sym));
                        }
                        Token::Ident(alias) => {
                            self.eat(&Token::Assign)?;
                            let Token::Str(orig) = self.next() else {
                                return Err(StarlarkError(
                                    "Expected string literal in load alias".to_string(),
                                ));
                            };
                            symbols.push((alias, orig));
                        }
                        other => {
                            return Err(StarlarkError(format!(
                                "Unexpected token in load(): {other:?}"
                            )))
                        }
                    }
                    if matches!(self.peek(), Token::Comma) {
                        self.next();
                    }
                }
                self.eat(&Token::RParen)?;
                Ok(Stmt::Load(LoadStatement {
                    module_label,
                    symbols,
                }))
            }
            _ => {
                let expr = self.parse_expr()?;
                if matches!(self.peek(), Token::Comma) {
                    if let Expr::Ident(first_var) = expr {
                        let mut targets = vec![first_var];
                        while matches!(self.peek(), Token::Comma) {
                            self.next();
                            if let Token::Ident(next_var) = self.next() {
                                targets.push(next_var);
                            }
                        }
                        self.eat(&Token::Assign)?;
                        let rhs = self.parse_expr()?;
                        return Ok(Stmt::TupleAssign {
                            targets,
                            expr: rhs,
                        });
                    }
                }
                if matches!(self.peek(), Token::Assign | Token::PlusAssign) {
                    let op_add = matches!(self.next(), Token::PlusAssign);
                    if let Expr::Ident(target) = expr {
                        let rhs = self.parse_expr()?;
                        return Ok(Stmt::Assign {
                            target,
                            op_add,
                            expr: rhs,
                        });
                    }
                }
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_var_list(&mut self) -> Result<Vec<String>, StarlarkError> {
        let mut vars = Vec::new();
        let parens = if matches!(self.peek(), Token::LParen) {
            self.next();
            true
        } else {
            false
        };
        let Token::Ident(first) = self.next() else {
            return Err(StarlarkError("Expected variable name".to_string()));
        };
        vars.push(first);
        while matches!(self.peek(), Token::Comma) {
            self.next();
            if let Token::Ident(v) = self.peek().clone() {
                if v == "in" {
                    break;
                }
                self.next();
                vars.push(v);
            } else {
                break;
            }
        }
        if parens {
            self.eat(&Token::RParen)?;
        }
        Ok(vars)
    }

    fn parse_expr(&mut self) -> Result<Expr, StarlarkError> {
        let expr = self.parse_or()?;
        if let Token::Ident(ref kw) = self.peek().clone() {
            if kw == "if" {
                self.next();
                let cond = self.parse_or()?;
                let Token::Ident(else_kw) = self.next() else {
                    return Err(StarlarkError("Expected 'else' in conditional expression".to_string()));
                };
                if else_kw != "else" {
                    return Err(StarlarkError("Expected 'else' in conditional expression".to_string()));
                }
                let else_body = self.parse_expr()?;
                return Ok(Expr::IfExpr {
                    body: Box::new(expr),
                    cond: Box::new(cond),
                    else_body: Box::new(else_body),
                });
            }
        }
        Ok(expr)
    }

    fn parse_or(&mut self) -> Result<Expr, StarlarkError> {
        let mut lhs = self.parse_and()?;
        while let Token::Ident(ref kw) = self.peek().clone() {
            if kw == "or" {
                self.next();
                let rhs = self.parse_and()?;
                lhs = Expr::BinOp {
                    op: BinOp::Or,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                };
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr, StarlarkError> {
        let mut lhs = self.parse_not()?;
        while let Token::Ident(ref kw) = self.peek().clone() {
            if kw == "and" {
                self.next();
                let rhs = self.parse_not()?;
                lhs = Expr::BinOp {
                    op: BinOp::And,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                };
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_not(&mut self) -> Result<Expr, StarlarkError> {
        if let Token::Ident(ref kw) = self.peek().clone() {
            if kw == "not" {
                self.next();
                let inner = self.parse_not()?;
                return Ok(Expr::UnaryNot(Box::new(inner)));
            }
        }
        self.parse_cmp()
    }

    fn parse_cmp(&mut self) -> Result<Expr, StarlarkError> {
        let mut lhs = self.parse_bitor()?;
        loop {
            let op = match self.peek().clone() {
                Token::EqEq => {
                    self.next();
                    Some(BinOp::Eq)
                }
                Token::NotEq => {
                    self.next();
                    Some(BinOp::Ne)
                }
                Token::Lt => {
                    self.next();
                    Some(BinOp::Lt)
                }
                Token::Le => {
                    self.next();
                    Some(BinOp::Le)
                }
                Token::Gt => {
                    self.next();
                    Some(BinOp::Gt)
                }
                Token::Ge => {
                    self.next();
                    Some(BinOp::Ge)
                }
                Token::Ident(ref kw) if kw == "in" => {
                    self.next();
                    Some(BinOp::In)
                }
                Token::Ident(ref kw) if kw == "not" => {
                    self.next();
                    let Token::Ident(in_kw) = self.next() else {
                        return Err(StarlarkError("Expected 'in' after 'not'".to_string()));
                    };
                    if in_kw != "in" {
                        return Err(StarlarkError("Expected 'in' after 'not'".to_string()));
                    }
                    Some(BinOp::NotIn)
                }
                _ => None,
            };
            if let Some(op) = op {
                let rhs = self.parse_bitor()?;
                lhs = Expr::BinOp {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                };
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_bitor(&mut self) -> Result<Expr, StarlarkError> {
        let mut lhs = self.parse_add()?;
        while matches!(self.peek(), Token::Pipe) {
            self.next();
            let rhs = self.parse_add()?;
            lhs = Expr::BinOp {
                op: BinOp::BitOr,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_add(&mut self) -> Result<Expr, StarlarkError> {
        let mut lhs = self.parse_mul()?;
        while matches!(self.peek(), Token::Plus | Token::Minus) {
            let op = match self.next() {
                Token::Plus => BinOp::Add,
                _ => BinOp::Sub,
            };
            let rhs = self.parse_mul()?;
            lhs = Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> Result<Expr, StarlarkError> {
        let mut lhs = self.parse_unary()?;
        while matches!(
            self.peek(),
            Token::Star | Token::Slash | Token::SlashSlash | Token::Percent
        ) {
            let op = match self.next() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::SlashSlash => BinOp::FloorDiv,
                _ => BinOp::Mod,
            };
            let rhs = self.parse_unary()?;
            lhs = Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, StarlarkError> {
        if matches!(self.peek(), Token::Minus) {
            self.next();
            let inner = self.parse_unary()?;
            return Ok(Expr::UnaryNeg(Box::new(inner)));
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr, StarlarkError> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek().clone() {
                Token::LParen => {
                    self.next();
                    let mut pos_args = Vec::new();
                    let mut kw_args = Vec::new();
                    while !matches!(self.peek(), Token::RParen | Token::Eof) {
                        if let Token::Ident(name) = self.peek().clone() {
                            if self.tokens.get(self.pos + 1) == Some(&Token::Assign) {
                                self.next(); // ident
                                self.next(); // =
                                let val = self.parse_expr()?;
                                kw_args.push((name, val));
                                if matches!(self.peek(), Token::Comma) {
                                    self.next();
                                }
                                continue;
                            }
                        }
                        if matches!(self.peek(), Token::Star | Token::StarStar) {
                            self.next();
                            let _ = self.parse_expr()?;
                        } else {
                            pos_args.push(self.parse_expr()?);
                        }
                        if matches!(self.peek(), Token::Comma) {
                            self.next();
                        }
                    }
                    self.eat(&Token::RParen)?;
                    expr = Expr::Call {
                        callee: Box::new(expr),
                        pos_args,
                        kw_args,
                    };
                }
                Token::Dot => {
                    self.next();
                    let Token::Ident(attr) = self.next() else {
                        return Err(StarlarkError("Expected attribute name after '.'".to_string()));
                    };
                    expr = Expr::Attr {
                        target: Box::new(expr),
                        attr,
                    };
                }
                Token::LBracket => {
                    self.next();
                    if matches!(self.peek(), Token::Colon) {
                        self.next();
                        let end = if matches!(self.peek(), Token::RBracket) {
                            None
                        } else {
                            Some(Box::new(self.parse_expr()?))
                        };
                        self.eat(&Token::RBracket)?;
                        expr = Expr::Slice {
                            target: Box::new(expr),
                            start: None,
                            end,
                        };
                    } else {
                        let first = self.parse_expr()?;
                        if matches!(self.peek(), Token::Colon) {
                            self.next();
                            let end = if matches!(self.peek(), Token::RBracket) {
                                None
                            } else {
                                Some(Box::new(self.parse_expr()?))
                            };
                            self.eat(&Token::RBracket)?;
                            expr = Expr::Slice {
                                target: Box::new(expr),
                                start: Some(Box::new(first)),
                                end,
                            };
                        } else {
                            self.eat(&Token::RBracket)?;
                            expr = Expr::Index {
                                target: Box::new(expr),
                                index: Box::new(first),
                            };
                        }
                    }
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, StarlarkError> {
        match self.next() {
            Token::Int(i) => Ok(Expr::Literal(Value::Int(i))),
            Token::Str(s) => Ok(Expr::Literal(Value::String(s))),
            Token::Ident(name) => match name.as_str() {
                "True" => Ok(Expr::Literal(Value::Bool(true))),
                "False" => Ok(Expr::Literal(Value::Bool(false))),
                "None" => Ok(Expr::Literal(Value::None)),
                _ => Ok(Expr::Ident(name)),
            },
            Token::LParen => {
                if matches!(self.peek(), Token::RParen) {
                    self.next();
                    return Ok(Expr::Tuple(Vec::new()));
                }
                let first = self.parse_expr()?;
                if matches!(self.peek(), Token::Comma) {
                    let mut items = vec![first];
                    while matches!(self.peek(), Token::Comma) {
                        self.next();
                        if matches!(self.peek(), Token::RParen) {
                            break;
                        }
                        items.push(self.parse_expr()?);
                    }
                    self.eat(&Token::RParen)?;
                    Ok(Expr::Tuple(items))
                } else {
                    self.eat(&Token::RParen)?;
                    Ok(first)
                }
            }
            Token::LBracket => {
                if matches!(self.peek(), Token::RBracket) {
                    self.next();
                    return Ok(Expr::List(Vec::new()));
                }
                let first = self.parse_expr()?;
                if let Token::Ident(ref kw) = self.peek().clone() {
                    if kw == "for" {
                        self.next();
                        let var_names = self.parse_var_list()?;
                        let Token::Ident(in_kw) = self.next() else {
                            return Err(StarlarkError(
                                "Expected 'in' in list comprehension".to_string(),
                            ));
                        };
                        if in_kw != "in" {
                            return Err(StarlarkError(
                                "Expected 'in' in list comprehension".to_string(),
                            ));
                        }
                        let iter = self.parse_or()?;
                        let cond = if let Token::Ident(ref if_kw) = self.peek().clone() {
                            if if_kw == "if" {
                                self.next();
                                Some(Box::new(self.parse_or()?))
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        self.eat(&Token::RBracket)?;
                        return Ok(Expr::ListComp {
                            body: Box::new(first),
                            var_names,
                            iter: Box::new(iter),
                            cond,
                        });
                    }
                }
                let mut items = vec![first];
                while matches!(self.peek(), Token::Comma) {
                    self.next();
                    if matches!(self.peek(), Token::RBracket) {
                        break;
                    }
                    items.push(self.parse_expr()?);
                }
                self.eat(&Token::RBracket)?;
                Ok(Expr::List(items))
            }
            Token::LBrace => {
                if matches!(self.peek(), Token::RBrace) {
                    self.next();
                    return Ok(Expr::Dict(Vec::new()));
                }
                let k = self.parse_expr()?;
                self.eat(&Token::Colon)?;
                let v = self.parse_expr()?;
                if let Token::Ident(ref kw) = self.peek().clone() {
                    if kw == "for" {
                        self.next();
                        let var_names = self.parse_var_list()?;
                        let Token::Ident(in_kw) = self.next() else {
                            return Err(StarlarkError(
                                "Expected 'in' in dict comprehension".to_string(),
                            ));
                        };
                        if in_kw != "in" {
                            return Err(StarlarkError(
                                "Expected 'in' in dict comprehension".to_string(),
                            ));
                        }
                        let iter = self.parse_or()?;
                        let cond = if let Token::Ident(ref if_kw) = self.peek().clone() {
                            if if_kw == "if" {
                                self.next();
                                Some(Box::new(self.parse_or()?))
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        self.eat(&Token::RBrace)?;
                        return Ok(Expr::DictComp {
                            key: Box::new(k),
                            val: Box::new(v),
                            var_names,
                            iter: Box::new(iter),
                            cond,
                        });
                    }
                }
                let mut pairs = vec![(k, v)];
                while matches!(self.peek(), Token::Comma) {
                    self.next();
                    if matches!(self.peek(), Token::RBrace) {
                        break;
                    }
                    let nk = self.parse_expr()?;
                    self.eat(&Token::Colon)?;
                    let nv = self.parse_expr()?;
                    pairs.push((nk, nv));
                }
                self.eat(&Token::RBrace)?;
                Ok(Expr::Dict(pairs))
            }
            other => Err(StarlarkError(format!(
                "Unexpected token in expression: {other:?}"
            ))),
        }
    }
}

/// Evaluation context for a Starlark module (`BUILD`, `.bzl`, or `MODULE.bazel`).
pub struct EvalContext {
    pub globals: BTreeMap<String, Value>,
    pub rule_calls: Vec<RecordedRuleCall>,
    pub loads: Vec<LoadStatement>,
    pub exports_files: Vec<String>,
    pub package_dir: PathFragment,
    pub fs: Option<Arc<dyn FileSystem>>,
}

impl Default for EvalContext {
    fn default() -> Self {
        Self::new(PathFragment::empty(), None)
    }
}

impl EvalContext {
    pub fn new(package_dir: PathFragment, fs: Option<Arc<dyn FileSystem>>) -> Self {
        Self {
            globals: BTreeMap::new(),
            rule_calls: Vec::new(),
            loads: Vec::new(),
            exports_files: Vec::new(),
            package_dir,
            fs,
        }
    }

    pub fn eval_str(&mut self, source: &str) -> Result<(), StarlarkError> {
        let tokens = lex(source)?;
        let mut parser = Parser::new(tokens);
        let stmts = parser.parse_module()?;
        for stmt in &stmts {
            let _ = self.exec_stmt(stmt)?;
        }
        Ok(())
    }

    fn exec_stmt(&mut self, stmt: &Stmt) -> Result<Option<Value>, StarlarkError> {
        match stmt {
            Stmt::Pass => Ok(None),
            Stmt::Load(load) => {
                self.loads.push(load.clone());
                for (local_name, exported_name) in &load.symbols {
                    if !self.globals.contains_key(local_name) {
                        self.globals.insert(
                            local_name.clone(),
                            Value::RuleCallable {
                                rule_class: exported_name.clone(),
                                executable: exported_name.ends_with("_binary"),
                                test: exported_name.ends_with("_test"),
                            },
                        );
                    }
                }
                Ok(None)
            }
            Stmt::Assign {
                target,
                op_add,
                expr,
            } => {
                let val = self.eval_expr(expr)?;
                if *op_add {
                    let prev = self
                        .globals
                        .get(target)
                        .cloned()
                        .ok_or_else(|| StarlarkError(format!("name '{target}' is not defined")))?;
                    let next_val = prev.add(&val)?;
                    self.globals.insert(target.clone(), next_val);
                } else {
                    let bound = match val {
                        Value::RuleCallable {
                            executable, test, ..
                        } => Value::RuleCallable {
                            rule_class: target.clone(),
                            executable,
                            test,
                        },
                        other => other,
                    };
                    self.globals.insert(target.clone(), bound);
                }
                Ok(None)
            }
            Stmt::TupleAssign { targets, expr } => {
                let val = self.eval_expr(expr)?;
                let items = match val {
                    Value::List(v) | Value::Tuple(v) => v,
                    _ => {
                        return Err(StarlarkError(
                            "cannot unpack non-iterable value".to_string(),
                        ))
                    }
                };
                for (i, t) in targets.iter().enumerate() {
                    if let Some(item) = items.get(i) {
                        self.globals.insert(t.clone(), item.clone());
                    }
                }
                Ok(None)
            }
            Stmt::Expr(expr) => {
                let _ = self.eval_expr(expr)?;
                Ok(None)
            }
            Stmt::Def(func) => {
                self.globals
                    .insert(func.name.clone(), Value::UserFunction(Arc::clone(func)));
                Ok(None)
            }
            Stmt::If {
                branches,
                else_body,
            } => {
                for (cond, body) in branches {
                    if self.eval_expr(cond)?.is_truthy() {
                        for s in body {
                            if let Some(ret) = self.exec_stmt(s)? {
                                return Ok(Some(ret));
                            }
                        }
                        return Ok(None);
                    }
                }
                for s in else_body {
                    if let Some(ret) = self.exec_stmt(s)? {
                        return Ok(Some(ret));
                    }
                }
                Ok(None)
            }
            Stmt::For {
                var_names,
                iter,
                body,
            } => {
                let iter_val = self.eval_expr(iter)?;
                let elements = Self::value_to_iter(iter_val)?;
                for elem in elements {
                    self.bind_loop_vars(var_names, elem)?;
                    for s in body {
                        if let Some(ret) = self.exec_stmt(s)? {
                            return Ok(Some(ret));
                        }
                    }
                }
                Ok(None)
            }
            Stmt::Return(opt_expr) => {
                let val = match opt_expr {
                    Some(e) => self.eval_expr(e)?,
                    None => Value::None,
                };
                Ok(Some(val))
            }
        }
    }

    fn value_to_iter(val: Value) -> Result<Vec<Value>, StarlarkError> {
        match val {
            Value::List(v) | Value::Tuple(v) => Ok(v),
            Value::Dict(d) => Ok(d.keys().cloned().map(Value::String).collect()),
            Value::Depset(d) => Ok(d.to_list()),
            other => Err(StarlarkError(format!(
                "type '{}' is not iterable",
                other.type_name()
            ))),
        }
    }

    fn bind_loop_vars(&mut self, var_names: &[String], elem: Value) -> Result<(), StarlarkError> {
        if var_names.len() == 1 {
            self.globals.insert(var_names[0].clone(), elem);
            return Ok(());
        }
        match elem {
            Value::List(items) | Value::Tuple(items) => {
                for (v, item) in var_names.iter().zip(items.into_iter()) {
                    self.globals.insert(v.clone(), item);
                }
                Ok(())
            }
            _ => Err(StarlarkError(
                "cannot unpack non-sequence in loop".to_string(),
            )),
        }
    }

    pub fn eval_expr(&mut self, expr: &Expr) -> Result<Value, StarlarkError> {
        match expr {
            Expr::Literal(v) => Ok(v.clone()),
            Expr::Ident(name) => {
                if let Some(v) = self.globals.get(name) {
                    return Ok(v.clone());
                }
                Ok(Value::RuleCallable {
                    rule_class: name.clone(),
                    executable: name.ends_with("_binary"),
                    test: name.ends_with("_test"),
                })
            }
            Expr::List(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.eval_expr(item)?);
                }
                Ok(Value::List(out))
            }
            Expr::Tuple(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.eval_expr(item)?);
                }
                Ok(Value::Tuple(out))
            }
            Expr::Dict(pairs) => {
                let mut map = BTreeMap::new();
                for (k_expr, v_expr) in pairs {
                    let k = self.eval_expr(k_expr)?.to_string();
                    let v = self.eval_expr(v_expr)?;
                    map.insert(k, v);
                }
                Ok(Value::Dict(map))
            }
            Expr::ListComp {
                body,
                var_names,
                iter,
                cond,
            } => {
                let iter_val = self.eval_expr(iter)?;
                let elements = Self::value_to_iter(iter_val)?;
                let mut out = Vec::new();
                for elem in elements {
                    self.bind_loop_vars(var_names, elem)?;
                    if let Some(ref c) = cond {
                        if !self.eval_expr(c)?.is_truthy() {
                            continue;
                        }
                    }
                    out.push(self.eval_expr(body)?);
                }
                Ok(Value::List(out))
            }
            Expr::DictComp {
                key,
                val,
                var_names,
                iter,
                cond,
            } => {
                let iter_val = self.eval_expr(iter)?;
                let elements = Self::value_to_iter(iter_val)?;
                let mut map = BTreeMap::new();
                for elem in elements {
                    self.bind_loop_vars(var_names, elem)?;
                    if let Some(ref c) = cond {
                        if !self.eval_expr(c)?.is_truthy() {
                            continue;
                        }
                    }
                    let k = self.eval_expr(key)?.to_string();
                    let v = self.eval_expr(val)?;
                    map.insert(k, v);
                }
                Ok(Value::Dict(map))
            }
            Expr::UnaryNot(inner) => Ok(Value::Bool(!self.eval_expr(inner)?.is_truthy())),
            Expr::UnaryNeg(inner) => match self.eval_expr(inner)? {
                Value::Int(i) => Ok(Value::Int(-i)),
                other => Err(StarlarkError(format!(
                    "unsupported unary -: {}",
                    other.type_name()
                ))),
            },
            Expr::IfExpr {
                body,
                cond,
                else_body,
            } => {
                if self.eval_expr(cond)?.is_truthy() {
                    self.eval_expr(body)
                } else {
                    self.eval_expr(else_body)
                }
            }
            Expr::BinOp { op, lhs, rhs } => {
                if *op == BinOp::And {
                    let l = self.eval_expr(lhs)?;
                    if !l.is_truthy() {
                        return Ok(l);
                    }
                    return self.eval_expr(rhs);
                }
                if *op == BinOp::Or {
                    let l = self.eval_expr(lhs)?;
                    if l.is_truthy() {
                        return Ok(l);
                    }
                    return self.eval_expr(rhs);
                }
                let l = self.eval_expr(lhs)?;
                let r = self.eval_expr(rhs)?;
                match op {
                    BinOp::Add | BinOp::BitOr => l.add(&r),
                    BinOp::Sub => match (l, r) {
                        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_sub(b))),
                        _ => Err(StarlarkError("unsupported '-' operands".to_string())),
                    },
                    BinOp::Mul => match (l, r) {
                        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_mul(b))),
                        (Value::String(s), Value::Int(n)) | (Value::Int(n), Value::String(s)) => {
                            Ok(Value::String(s.repeat(n.max(0) as usize)))
                        }
                        _ => Err(StarlarkError("unsupported '*' operands".to_string())),
                    },
                    BinOp::Div | BinOp::FloorDiv => match (l, r) {
                        (Value::Int(a), Value::Int(b)) if b != 0 => Ok(Value::Int(a / b)),
                        _ => Err(StarlarkError("division error".to_string())),
                    },
                    BinOp::Mod => match (l, r) {
                        (Value::Int(a), Value::Int(b)) if b != 0 => Ok(Value::Int(a % b)),
                        (Value::String(fmt_str), Value::Tuple(args)) => {
                            Ok(Value::String(percent_format(&fmt_str, &args)))
                        }
                        (Value::String(fmt_str), single) => {
                            Ok(Value::String(percent_format(&fmt_str, &[single])))
                        }
                        _ => Err(StarlarkError("modulo error".to_string())),
                    },
                    BinOp::Eq => Ok(Value::Bool(l == r)),
                    BinOp::Ne => Ok(Value::Bool(l != r)),
                    BinOp::Lt => match (l, r) {
                        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
                        (Value::String(a), Value::String(b)) => Ok(Value::Bool(a < b)),
                        _ => Ok(Value::Bool(false)),
                    },
                    BinOp::Le => match (l, r) {
                        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
                        (Value::String(a), Value::String(b)) => Ok(Value::Bool(a <= b)),
                        _ => Ok(Value::Bool(false)),
                    },
                    BinOp::Gt => match (l, r) {
                        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
                        (Value::String(a), Value::String(b)) => Ok(Value::Bool(a > b)),
                        _ => Ok(Value::Bool(false)),
                    },
                    BinOp::Ge => match (l, r) {
                        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
                        (Value::String(a), Value::String(b)) => Ok(Value::Bool(a >= b)),
                        _ => Ok(Value::Bool(false)),
                    },
                    BinOp::In => Ok(Value::Bool(value_contains(&r, &l))),
                    BinOp::NotIn => Ok(Value::Bool(!value_contains(&r, &l))),
                    BinOp::And | BinOp::Or => unreachable!(),
                }
            }
            Expr::Attr { target, attr } => {
                let recv = self.eval_expr(target)?;
                match recv {
                    Value::Struct(fields)
                    | Value::ProviderInstance { fields, .. }
                    | Value::Dict(fields) => Ok(fields.get(attr).cloned().unwrap_or(Value::None)),
                    Value::Label(l) => match attr.as_str() {
                        "name" => Ok(Value::String(l.name().to_string())),
                        "package" => Ok(Value::String(l.package_name().to_string())),
                        "repo_name" | "workspace_name" => {
                            Ok(Value::String(l.repository().name().to_string()))
                        }
                        _ => Ok(Value::None),
                    },
                    _ => Ok(Value::None),
                }
            }
            Expr::Index { target, index } => {
                let recv = self.eval_expr(target)?;
                let idx = self.eval_expr(index)?;
                match (recv, idx) {
                    (Value::List(v) | Value::Tuple(v), Value::Int(i)) => {
                        let actual = if i < 0 {
                            (v.len() as isize + i as isize) as usize
                        } else {
                            i as usize
                        };
                        v.get(actual)
                            .cloned()
                            .ok_or_else(|| StarlarkError("index out of range".to_string()))
                    }
                    (Value::Dict(d), k) => d
                        .get(&k.to_string())
                        .cloned()
                        .ok_or_else(|| StarlarkError(format!("key '{k}' not found in dict"))),
                    _ => Ok(Value::None),
                }
            }
            Expr::Slice { target, start, end } => {
                let recv = self.eval_expr(target)?;
                let s_idx = match start {
                    Some(e) => match self.eval_expr(e)? {
                        Value::Int(i) => i.max(0) as usize,
                        _ => 0,
                    },
                    None => 0,
                };
                match recv {
                    Value::List(v) => {
                        let e_idx = match end {
                            Some(e) => match self.eval_expr(e)? {
                                Value::Int(i) => (i.max(0) as usize).min(v.len()),
                                _ => v.len(),
                            },
                            None => v.len(),
                        };
                        Ok(Value::List(v[s_idx.min(e_idx)..e_idx].to_vec()))
                    }
                    Value::String(s) => {
                        let e_idx = match end {
                            Some(e) => match self.eval_expr(e)? {
                                Value::Int(i) => (i.max(0) as usize).min(s.len()),
                                _ => s.len(),
                            },
                            None => s.len(),
                        };
                        Ok(Value::String(s[s_idx.min(e_idx)..e_idx].to_string()))
                    }
                    _ => Ok(Value::None),
                }
            }
            Expr::Call {
                callee,
                pos_args,
                kw_args,
            } => self.eval_call(callee, pos_args, kw_args),
        }
    }

    fn eval_call(
        &mut self,
        callee: &Expr,
        pos_args: &[Expr],
        kw_args: &[(String, Expr)],
    ) -> Result<Value, StarlarkError> {
        // Method call on receiver (`obj.method(...)`)
        if let Expr::Attr { target, attr } = callee {
            let recv = self.eval_expr(target)?;
            let mut eval_pos = Vec::new();
            for a in pos_args {
                eval_pos.push(self.eval_expr(a)?);
            }
            let mut eval_kw = BTreeMap::new();
            for (k, v) in kw_args {
                eval_kw.insert(k.clone(), self.eval_expr(v)?);
            }
            return self.eval_method_call(recv, attr, &eval_pos, &eval_kw);
        }

        let mut eval_pos = Vec::new();
        for a in pos_args {
            eval_pos.push(self.eval_expr(a)?);
        }
        let mut eval_kw = BTreeMap::new();
        for (k, v) in kw_args {
            eval_kw.insert(k.clone(), self.eval_expr(v)?);
        }

        if let Expr::Ident(func_name) = callee {
            match func_name.as_str() {
                "select" => {
                    let Some(Value::Dict(map)) = eval_pos.first() else {
                        return Err(StarlarkError("select() requires a dict argument".to_string()));
                    };
                    return Ok(Value::SelectorList(SelectorList::from_select(map.clone())));
                }
                "depset" => {
                    let direct = eval_pos
                        .first()
                        .or_else(|| eval_kw.get("direct"))
                        .and_then(|v| match v {
                            Value::List(l) | Value::Tuple(l) => Some(l.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let order = eval_kw
                        .get("order")
                        .and_then(|v| v.as_str())
                        .map(DepsetOrder::parse)
                        .transpose()?
                        .unwrap_or_default();
                    let transitive = eval_kw
                        .get("transitive")
                        .and_then(|v| match v {
                            Value::List(l) => Some(
                                l.iter()
                                    .filter_map(|item| match item {
                                        Value::Depset(d) => Some(Arc::clone(d)),
                                        _ => None,
                                    })
                                    .collect(),
                            ),
                            _ => None,
                        })
                        .unwrap_or_default();
                    return Ok(Value::Depset(Arc::new(Depset::new(
                        order, direct, transitive,
                    )?)));
                }
                "glob" => {
                    let include = eval_pos
                        .first()
                        .or_else(|| eval_kw.get("include"))
                        .map(|v| v.to_string_list())
                        .unwrap_or_default();
                    let exclude = eval_pos
                        .get(1)
                        .or_else(|| eval_kw.get("exclude"))
                        .map(|v| v.to_string_list())
                        .unwrap_or_default();
                    let exclude_dirs = eval_kw
                        .get("exclude_directories")
                        .map(|v| match v {
                            Value::Int(i) => *i != 0,
                            Value::Bool(b) => *b,
                            _ => true,
                        })
                        .unwrap_or(true);
                    if let Some(ref fs) = self.fs {
                        if let Ok(matched) = evaluate_glob(
                            fs.as_ref(),
                            &self.package_dir,
                            &include,
                            &exclude,
                            exclude_dirs,
                        ) {
                            return Ok(Value::List(
                                matched.into_iter().map(Value::String).collect(),
                            ));
                        }
                    }
                    return Ok(Value::List(Vec::new()));
                }
                "struct" => {
                    return Ok(Value::Struct(eval_kw));
                }
                "provider" => {
                    return Ok(Value::Provider {
                        name: "Provider".to_string(),
                    });
                }
                "rule" | "repository_rule" | "module_extension" | "aspect" | "tag_class" => {
                    let executable = eval_kw
                        .get("executable")
                        .map(|v| v.is_truthy())
                        .unwrap_or(false);
                    let test = eval_kw.get("test").map(|v| v.is_truthy()).unwrap_or(false);
                    return Ok(Value::RuleCallable {
                        rule_class: "custom_rule".to_string(),
                        executable,
                        test,
                    });
                }
                "Label" => {
                    let s = eval_pos
                        .first()
                        .and_then(|v| v.as_str())
                        .unwrap_or("//:default");
                    let label = Label::parse_canonical(s)
                        .map_err(|e| StarlarkError(e.to_string()))?;
                    return Ok(Value::Label(label));
                }
                "exports_files" => {
                    if let Some(first) = eval_pos.first() {
                        self.exports_files.extend(first.to_string_list());
                    }
                    return Ok(Value::None);
                }
                "package" | "licenses" | "register_toolchains" | "register_execution_platforms" => {
                    return Ok(Value::None);
                }
                "len" => {
                    let n = match eval_pos.first() {
                        Some(Value::String(s)) => s.len() as i64,
                        Some(Value::List(l) | Value::Tuple(l)) => l.len() as i64,
                        Some(Value::Dict(d) | Value::Struct(d)) => d.len() as i64,
                        _ => 0,
                    };
                    return Ok(Value::Int(n));
                }
                "range" => {
                    let (start, end, step) = match eval_pos.as_slice() {
                        [Value::Int(e)] => (0, *e, 1),
                        [Value::Int(s), Value::Int(e)] => (*s, *e, 1),
                        [Value::Int(s), Value::Int(e), Value::Int(st)] => (*s, *e, *st),
                        _ => (0, 0, 1),
                    };
                    let mut out = Vec::new();
                    let mut cur = start;
                    while (step > 0 && cur < end) || (step < 0 && cur > end) {
                        out.push(Value::Int(cur));
                        cur += step;
                    }
                    return Ok(Value::List(out));
                }
                "str" | "repr" => {
                    let s = eval_pos
                        .first()
                        .map(|v| v.to_string())
                        .unwrap_or_default();
                    return Ok(Value::String(s));
                }
                "int" => {
                    let i = match eval_pos.first() {
                        Some(Value::Int(n)) => *n,
                        Some(Value::String(s)) => s.trim().parse().unwrap_or(0),
                        Some(Value::Bool(b)) => i64::from(*b),
                        _ => 0,
                    };
                    return Ok(Value::Int(i));
                }
                "bool" => {
                    let b = eval_pos.first().map(|v| v.is_truthy()).unwrap_or(false);
                    return Ok(Value::Bool(b));
                }
                "fail" => {
                    let msg = eval_pos
                        .first()
                        .or_else(|| eval_kw.get("msg"))
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "fail() called".to_string());
                    return Err(StarlarkError(msg));
                }
                _ => {}
            }
        }

        let callee_val = self.eval_expr(callee)?;
        match callee_val {
            Value::UserFunction(func) => {
                let saved_globals = self.globals.clone();
                for (i, param) in func.params.iter().enumerate() {
                    if param.is_args {
                        let rest = if i < eval_pos.len() {
                            eval_pos[i..].to_vec()
                        } else {
                            Vec::new()
                        };
                        self.globals.insert(param.name.clone(), Value::Tuple(rest));
                    } else if param.is_kwargs {
                        self.globals
                            .insert(param.name.clone(), Value::Dict(eval_kw.clone()));
                    } else if let Some(val) = eval_pos.get(i) {
                        self.globals.insert(param.name.clone(), val.clone());
                    } else if let Some(val) = eval_kw.get(&param.name) {
                        self.globals.insert(param.name.clone(), val.clone());
                    } else if let Some(ref def_expr) = param.default {
                        let def_val = self.eval_expr(def_expr)?;
                        self.globals.insert(param.name.clone(), def_val);
                    } else {
                        self.globals.insert(param.name.clone(), Value::None);
                    }
                }
                let mut ret = Value::None;
                for s in &func.body {
                    if let Some(r) = self.exec_stmt(s)? {
                        ret = r;
                        break;
                    }
                }
                self.globals = saved_globals;
                Ok(ret)
            }
            Value::Provider { name } => Ok(Value::ProviderInstance {
                provider_name: name,
                fields: eval_kw,
            }),
            Value::RuleCallable { rule_class, .. } => {
                self.rule_calls.push(RecordedRuleCall {
                    rule_class,
                    kwargs: eval_kw,
                });
                Ok(Value::None)
            }
            _ => Ok(Value::None),
        }
    }

    fn eval_method_call(
        &mut self,
        recv: Value,
        method: &str,
        pos_args: &[Value],
        _kw_args: &BTreeMap<String, Value>,
    ) -> Result<Value, StarlarkError> {
        match (recv, method) {
            (Value::Depset(d), "to_list") => Ok(Value::List(d.to_list())),
            (Value::String(s), "startswith") => {
                let prefix = pos_args.first().and_then(|v| v.as_str()).unwrap_or("");
                Ok(Value::Bool(s.starts_with(prefix)))
            }
            (Value::String(s), "endswith") => {
                let suffix = pos_args.first().and_then(|v| v.as_str()).unwrap_or("");
                Ok(Value::Bool(s.ends_with(suffix)))
            }
            (Value::String(s), "replace") => {
                let old = pos_args.first().and_then(|v| v.as_str()).unwrap_or("");
                let new = pos_args.get(1).and_then(|v| v.as_str()).unwrap_or("");
                Ok(Value::String(s.replace(old, new)))
            }
            (Value::String(s), "split") => {
                let sep = pos_args.first().and_then(|v| v.as_str()).unwrap_or(" ");
                Ok(Value::List(
                    s.split(sep).map(|p| Value::String(p.to_string())).collect(),
                ))
            }
            (Value::String(sep), "join") => {
                let items = pos_args
                    .first()
                    .map(|v| v.to_string_list())
                    .unwrap_or_default();
                Ok(Value::String(items.join(&sep)))
            }
            (Value::String(s), "strip") => Ok(Value::String(s.trim().to_string())),
            (Value::String(s), "upper") => Ok(Value::String(s.to_uppercase())),
            (Value::String(s), "lower") => Ok(Value::String(s.to_lowercase())),
            (Value::String(s), "format") => {
                let mut formatted = s;
                for (i, arg) in pos_args.iter().enumerate() {
                    formatted = formatted.replacen("{}", &arg.to_string(), 1);
                    formatted = formatted.replace(&format!("{{{i}}}"), &arg.to_string());
                }
                Ok(Value::String(formatted))
            }
            (Value::Dict(d), "items") => Ok(Value::List(
                d.into_iter()
                    .map(|(k, v)| Value::Tuple(vec![Value::String(k), v]))
                    .collect(),
            )),
            (Value::Dict(d), "keys") => {
                Ok(Value::List(d.into_keys().map(Value::String).collect()))
            }
            (Value::Dict(d), "values") => Ok(Value::List(d.into_values().collect())),
            (Value::Dict(d), "get") => {
                let k = pos_args.first().map(|v| v.to_string()).unwrap_or_default();
                let default = pos_args.get(1).cloned().unwrap_or(Value::None);
                Ok(d.get(&k).cloned().unwrap_or(default))
            }
            _ => Ok(Value::None),
        }
    }
}

fn value_contains(container: &Value, item: &Value) -> bool {
    match container {
        Value::List(v) | Value::Tuple(v) => v.contains(item),
        Value::Dict(d) => d.contains_key(&item.to_string()),
        Value::String(s) => item.as_str().map(|sub| s.contains(sub)).unwrap_or(false),
        _ => false,
    }
}

fn percent_format(fmt_str: &str, args: &[Value]) -> String {
    let mut out = fmt_str.to_string();
    for arg in args {
        if let Some(idx) = out.find("%s").or_else(|| out.find("%d")) {
            out.replace_range(idx..idx + 2, &arg.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use bazel_vfs::InMemoryFileSystem;

    #[test]
    fn test_starlark_depset_and_comprehensions() {
        let mut ctx = EvalContext::default();
        ctx.eval_str(
            r#"
d1 = depset(["a", "b"])
d2 = depset(["b", "c"], transitive = [d1])
items = [x.upper() for x in d2.to_list() if x != "b"]
"#,
        )
        .unwrap();
        assert_eq!(
            ctx.globals.get("items"),
            Some(&Value::List(vec![
                Value::String("A".to_string()),
                Value::String("C".to_string()),
            ]))
        );
    }

    #[test]
    fn test_starlark_macros_glob_and_select_in_build() {
        let fs = Arc::new(InMemoryFileSystem::default());
        let pkg_dir = PathFragment::create("/workspace/pkg");
        fs.create_directory_and_parents(&pkg_dir).unwrap();
        fs.write_file(&pkg_dir.get_relative_str("a.cc"), b"").unwrap();
        fs.write_file(&pkg_dir.get_relative_str("b.cc"), b"").unwrap();
        fs.write_file(&pkg_dir.get_relative_str("b_test.cc"), b"")
            .unwrap();

        let mut ctx = EvalContext::new(pkg_dir, Some(fs));
        ctx.eval_str(
            r#"
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

def my_cc_macro(name, extra_deps = []):
    cc_library(
        name = name + "_lib",
        srcs = glob(["*.cc"], exclude = ["*_test.cc"]),
        deps = extra_deps + select({
            "//src/conditions:darwin": ["//platform:mac"],
            "//conditions:default": ["//platform:generic"],
        }),
    )

my_cc_macro("engine", extra_deps = ["//base"])
"#,
        )
        .unwrap();

        assert_eq!(ctx.loads.len(), 1);
        assert_eq!(ctx.rule_calls.len(), 1);
        let call = &ctx.rule_calls[0];
        assert_eq!(call.rule_class, "cc_library");
        assert_eq!(
            call.kwargs.get("name"),
            Some(&Value::String("engine_lib".to_string()))
        );
        assert_eq!(
            call.kwargs.get("srcs").unwrap().to_string_list(),
            vec!["a.cc", "b.cc"]
        );
        assert_eq!(
            call.kwargs.get("deps").unwrap().to_string_list(),
            vec!["//base", "//platform:generic"]
        );
    }
}

