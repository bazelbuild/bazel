# Copyright 2026 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates Stage 1 Rust bootstrap crate manifests for bazel-distfile."""

load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load(
    "@rules_rust//rust:rust_common.bzl",
    "BuildInfo",
    "CrateGroupInfo",
    "CrateInfo",
    "TestCrateInfo",
)

_BOOTSTRAP_PLATFORMS = {
    "linux": Label("@llvm//platforms:linux_x86_64"),
    "darwin": Label("@llvm//platforms:macos_aarch64"),
    "windows": Label("//:windows_x86_64"),
}

_RustBootstrapInfo = provider(
    doc = "Transitive first-party Rust crates collected for Stage 1 bootstrap.",
    fields = {
        "crate": "Optional[struct]: Descriptor for this crate target.",
        "crates": "depset[struct]: Postorder transitive crate descriptors.",
        "group_crates": "list[struct]: Expanded crates for CrateGroupInfo targets.",
    },
)

def _sh_quote(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"

def _iter_attr_targets(rule_attr, attr_names):
    targets = []
    for name in attr_names:
        val = getattr(rule_attr, name, None)
        if val == None:
            continue
        if type(val) == "list":
            targets.extend(val)
        elif type(val) == "Target":
            targets.append(val)
    return targets

def _rust_bootstrap_aspect_impl(target, ctx):
    if BuildInfo in target:
        fail(
            "Stage 1 Rust bootstrap does not support cargo_build_script yet: {}".format(
                target.label,
            ),
        )

    if CrateGroupInfo in target:
        group_crates = []
        transitive = []
        for dep in getattr(ctx.rule.attr, "deps", []):
            if _RustBootstrapInfo not in dep:
                continue
            info = dep[_RustBootstrapInfo]
            transitive.append(info.crates)
            if info.crate:
                group_crates.append(info.crate)
            group_crates.extend(info.group_crates)
        return [_RustBootstrapInfo(
            crate = None,
            crates = depset(transitive = transitive, order = "postorder"),
            group_crates = group_crates,
        )]

    if CrateInfo in target:
        crate_info = target[CrateInfo]
    elif TestCrateInfo in target and target[TestCrateInfo].crate.type not in ("staticlib", "cdylib"):
        crate_info = target[TestCrateInfo].crate
    else:
        transitive_crates = [
            dep[_RustBootstrapInfo].crates
            for dep in _iter_attr_targets(ctx.rule.attr, ("deps", "runtime_deps", "resources", "src"))
            if _RustBootstrapInfo in dep
        ]
        if not transitive_crates:
            return []
        return [_RustBootstrapInfo(
            crate = None,
            crates = depset(transitive = transitive_crates, order = "postorder"),
            group_crates = [],
        )]

    if crate_info.owner.repo_name:
        fail(
            "Stage 1 Rust bootstrap does not support third-party crates yet: {}".format(
                crate_info.owner,
            ),
        )
    if crate_info.type == "proc-macro":
        fail(
            "Stage 1 Rust bootstrap does not support proc-macro crates yet: {}".format(
                crate_info.owner,
            ),
        )

    alias_map = {
        aliased_target.label: alias_name
        for aliased_target, alias_name in getattr(ctx.rule.attr, "aliases", {}).items()
    }

    externs = []
    transitive_crates = []
    for dep in getattr(ctx.rule.attr, "deps", []):
        if _RustBootstrapInfo not in dep:
            continue
        info = dep[_RustBootstrapInfo]
        transitive_crates.append(info.crates)
        if info.crate:
            externs.append((alias_map.get(dep.label, info.crate.name), info.crate.key))
        for gc in info.group_crates:
            externs.append((gc.name, gc.key))

    crate = struct(
        key = str(target.label),
        name = crate_info.name,
        edition = crate_info.edition,
        root = crate_info.root.short_path,
        features = tuple(getattr(ctx.rule.attr, "crate_features", [])),
        flags = tuple(getattr(ctx.rule.attr, "rustc_flags", [])),
        env = tuple(sorted(crate_info.rustc_env.items())),
        externs = tuple(externs),
    )

    return [_RustBootstrapInfo(
        crate = crate,
        crates = depset(direct = [crate], transitive = transitive_crates, order = "postorder"),
        group_crates = [],
    )]

_rust_bootstrap_aspect = aspect(
    implementation = _rust_bootstrap_aspect_impl,
    attr_aspects = ["deps", "runtime_deps", "resources", "src", "proc_macro_deps"],
    apply_to_generating_rules = True,
)

def _os_split_transition_impl(settings, attr):
    _ = settings, attr
    return {
        os_name: {"//command_line_option:platforms": [platform_label]}
        for os_name, platform_label in _BOOTSTRAP_PLATFORMS.items()
    }

_os_split_transition = transition(
    implementation = _os_split_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:platforms"],
)

def _render_manifest(crates):
    lines = ["# Auto-generated by Bazel (//src:bootstrap-rust). Do not edit."]
    rlib_by_key = {}
    for idx, crate in enumerate(crates):
        cid = "r{}".format(idx)
        rlib_by_key[crate.key] = "lib{}-{}.rlib".format(crate.name, cid)

        parts = [
            "rust_crate",
            _sh_quote(cid),
            _sh_quote(crate.name),
            _sh_quote(crate.edition),
            _sh_quote(crate.root),
        ]
        for k, v in crate.env:
            parts.extend(["--env", _sh_quote("{}={}".format(k, v))])
        for ext_name, dep_key in crate.externs:
            parts.extend(["--extern", _sh_quote("{}={}".format(ext_name, rlib_by_key[dep_key]))])

        extra_flags = []
        for feat in crate.features:
            extra_flags.extend(["--cfg", _sh_quote('feature="{}"'.format(feat))])
        for flag in crate.flags:
            extra_flags.append(_sh_quote(flag))
        if extra_flags:
            parts.append("--")
            parts.extend(extra_flags)

        lines.append(" ".join(parts))

    return "\n".join(lines) + "\n"

def _rust_bootstrap_manifests_impl(ctx):
    outputs = []
    for os_name in sorted(_BOOTSTRAP_PLATFORMS.keys()):
        crates = depset(
            transitive = [
                dep[_RustBootstrapInfo].crates
                for dep in ctx.split_attr.deps.get(os_name, [])
                if _RustBootstrapInfo in dep
            ],
            order = "postorder",
        ).to_list()
        out = ctx.actions.declare_file("{}.sh".format(os_name))
        ctx.actions.write(
            output = out,
            content = _render_manifest(crates),
        )
        outputs.append(out)

    return [DefaultInfo(files = depset(outputs))]

_rust_bootstrap_manifests = rule(
    implementation = _rust_bootstrap_manifests_impl,
    attrs = {
        "deps": attr.label_list(
            aspects = [_rust_bootstrap_aspect],
            cfg = _os_split_transition,
        ),
    },
)

def rust_bootstrap_bundle(name, deps, visibility = None):
    """Generates a tar archive of derived/rust_bootstrap/<os>.sh manifests."""
    manifests_name = name + "_manifests"
    _rust_bootstrap_manifests(
        name = manifests_name,
        deps = deps,
    )
    pkg_tar(
        name = name,
        srcs = [":" + manifests_name],
        mode = "0644",
        package_dir = "derived/rust_bootstrap",
        visibility = visibility,
    )
