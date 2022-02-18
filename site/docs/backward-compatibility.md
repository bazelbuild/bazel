---
layout: documentation
title: Backward compatibility
category: getting-started
---

<div style="background-color: #EFCBCB; color: #AE2B2B;  border: 1px solid #AE2B2B; border-radius: 5px; border-left: 10px solid #AE2B2B; padding: 0.5em;">
<b>IMPORTANT:</b> The Bazel docs have moved! Please update your bookmark to <a href="https://bazel.build/release/backward-compatibility" style="color: #0000EE;">https://bazel.build/release/backward-compatibility</a>
<p/>
You can <a href="https://blog.bazel.build/2022/02/17/Launching-new-Bazel-site.html" style="color: #0000EE;">read about</a> the migration, and let us <a href="https://forms.gle/onkAkr2ZwBmcbWXj7" style="color: #0000EE;">know what you think</a>.
</div>


# Backward Compatibility

This page provides information on how to handle backward compatibility,
including migrating from one release to another and how to communicate
incompatible changes.

Bazel is evolving. Minor versions released as part of an
[LTS major version](versioning.html#lts-releases) are fully backward-compatible.
Changes between major LTS releases may contain incompatible changes that require
some migration effort. For more information on how the Bazel release cadence
works, see
[Announcing Bazel Long Term Support (LTS) releases](https://blog.bazel.build/2020/11/10/long-term-support-release.html).

## At a glance

1. It is recommended to use `--incompatible_*` flags for breaking changes.
1. For every `--incompatible_*` flag, a GitHub issue explains
   the change in behavior and aims to provide a migration recipe.
1. APIs and behavior guarded by an `--experimental_*` flag can change at any time.
1. Never run production builds with `--experimental_*`  or `--incompatible_*` flags.

## How to follow this policy

* [For Bazel users - how to update Bazel](updating-bazel.html)
* [For contributors - best practices for incompatible changes](https://bazel.build/breaking-changes-guide.html)
* <a href='https://github.com/bazelbuild/continuous-integration/tree/master/docs/release-playbook.%6D%64'>For release managers - how to update issue labels and release</a>


## What is stable functionality?

In general, APIs or behaviors without `--experimental_...` flags are considered
stable, supported features in Bazel.

This includes:

* Starlark language and APIs
* Rules bundled with Bazel
* Bazel APIs such as Remote Execution APIs or Build Event Protocol
* Flags and their semantics

## Incompatible changes and migration recipes

For every incompatible change in a new release, the Bazel team aims to provide a
_migration recipe_ that helps you update your code
(`BUILD` and `.bzl` files, as well as any Bazel usage in scripts,
usage of Bazel API, and so on).

Incompatible changes should have an associated `--incompatible_*` flag and a
corresponding GitHub issue.

## Communicating incompatible changes

The primary source of information about incompatible changes are GitHub issues
marked with an ["incompatible-change" label](https://github.com/bazelbuild/bazel/issues?q=label%3Aincompatible-change).

For every incompatible change, the issue specifies the following:
* Name of the flag controlling the incompatible change
* Description of the changed functionality
* Migration recipe

The incompatible change issue is closed when the incompatible flag is flipped at
HEAD. All incompatible changes that are expected to happen in release X.Y
are marked with a label "breaking-change-X.Y"."



