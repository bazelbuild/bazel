---
layout: documentation
title: Backward Compatibility
---

# Backward Compatibility

Bazel is evolving, and we will make changes to Bazel that at times will be
incompatible and require some changes from Bazel users.

## GitHub labels

* All incompatible changes: label [**incompatible-change**](https://github.com/bazelbuild/bazel/issues?q=label%3Aincompatible-change)
* Expected breaking change in release X.Y: label **breaking-change-X.Y** (e.g. [**breaking-change-0.21**](https://github.com/bazelbuild/bazel/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3Abreaking-change-0.21))
* Release X.Y is in a migration window: label **migration-X.Y** (e.g. [**migration-0.21**](https://github.com/bazelbuild/bazel/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3Amigration-0.21))

## At a glance

1. Every breaking change is guarded with an `--incompatible_*` flag.
1. Newly introduced incompatible flags default to off.
1. For every `--incompatible_*` flag we have a GitHub issue explaining
   the change in behavior and giving a migration recipe.
1. The migration window is at least one release long and is set by the author of the incompatible change.
1. We announce what set of flags we indend to flip with the next release one release in advance.
1. APIs and behavior guarded by an `--experimental_*` flag can change at any time.
1. Users should never run their production builds with `--experimental_*`  or `--incompatible_*` flags.

## What is stable functionality?

In general, if an API or a behavior is available in Bazel without
`--experimental_...` flag, we consider it a stable, supported feature.
This includes:

* Starlark language and APIs
* Rules bundled with Bazel
* Bazel APIs such as Remote Execution APIs or Build Event Protocol
* Flags and their semantics

## Incompatible changes and migration recipes

When we introduce an incompatible change, we try to make it easier for Bazel
users to update their code. We do this by means of _migration windows_ and
_migration recipes_.

Migration window is one or more release of Bazel during which a migration from
old functionality to new functionality is possible, according to a migration
recipe.

During the migration window, both the old functionality and the new functionality
are available in the Bazel release. For every incompatible change, we provide
a _migration recipe_ that allows updating the user code (`BUILD` and `.bzl` files,
as well as any Bazel usage in scripts, usage of Bazel API and so on) in such a
way that **it works simultaneously without any flags with old and new
functionality**.

In other words, during a migration window for an incompatible change `foo`:

1. `--incompatible_foo` flag is available in Bazel release and defaults to off.
1. User code can be updated in such a way that it works simultaneously with
   that flag being on or off.
1. After the code is migrated, the users can check whether they migrated
   successfully by building with `--incompatible_foo=true`. However, their
   code will continue to work in the same release in default state (where
   `--incompatible_foo` is of), as well after the migration window is over
   (at which point the flag will be effectively on).

## Communicating incompatible changes

The primary source of information about incompatible changes are GitHub issues
marked with an ["incompatible-change" label](https://github.com/bazelbuild/bazel/issues?q=label%3Aincompatible-change).

For every incompatible change, the issue specifies:
* the name of the flag controlling the incompatible change
* the description of the changed functionality
* the migration recipe

The incompatible change issue is closed when the incompatible flag is flipped at
HEAD.

All the incompatible changes for which a Bazel releaze X.Y is part of a
migration window are marked with a label "migration-X.Y" label (for example
[migration-0.21](https://github.com/bazelbuild/bazel/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3Amigration-0.21)).

All the incompatible changes that are expected to happen in release X.Y
are marked with a label "breaking-change-X.Y" (for example
[breaking-change-0.21](https://github.com/bazelbuild/bazel/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3Abreaking-change-0.21)).


