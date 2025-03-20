Project: /_project.yaml
Book: /_book.yaml

# Backward Compatibility

{% include "_buttons.html" %}

This page provides information about how to handle backward compatibility,
including migrating from one release to another and how to communicate
incompatible changes.

Bazel is evolving. Minor versions released as part of an [LTS major
version](/release#bazel-versioning) are fully backward-compatible. New major LTS
releases may contain incompatible changes that require some migration effort.
For more information about Bazel's release model, please check out the [Release
Model](/release) page.

## Summary {:#summary}

1.  It is recommended to use `--incompatible_*` flags for breaking changes.
1.  For every `--incompatible_*` flag, a GitHub issue explains the change in
    behavior and aims to provide a migration recipe.
1.  Incompatible flags are recommended to be back-ported to the latest LTS
    release without enabling the flag by default.
1.  APIs and behavior guarded by an `--experimental_*` flag can change at any
    time.
1.  Never run production builds with `--experimental_*` or `--incompatible_*`
    flags.

## How to follow this policy {:#policy}

*   [For Bazel users - how to update Bazel](/install/bazelisk)
*   [For contributors - best practices for incompatible changes](/contribute/breaking-changes)
*   [For release managers - how to update issue labels and release](https://github.com/bazelbuild/continuous-integration/tree/master/docs/release-playbook.%6D%64){: .external}

## What is stable functionality? {:#stable-functionality}

In general, APIs or behaviors without `--experimental_...` flags are considered
stable, supported features in Bazel.

This includes:

*   Starlark language and APIs
*   Rules bundled with Bazel
*   Bazel APIs such as Remote Execution APIs or Build Event Protocol
*   Flags and their semantics

## Incompatible changes and migration recipes {:#incompatible-changes}

For every incompatible change in a new release, the Bazel team aims to provide a
_migration recipe_ that helps you update your code (`BUILD` and `.bzl` files, as
well as any Bazel usage in scripts, usage of Bazel API, and so on).

Incompatible changes should have an associated `--incompatible_*` flag and a
corresponding GitHub issue.

The incompatible flag and relevant changes are recommended to be back-ported to
the latest LTS release without enabling the flag by default. This allows users
to migrate for the incompatible changes before the next LTS release is
available.

## Communicating incompatible changes {:#communicating-incompatible-changes}

The primary source of information about incompatible changes are GitHub issues
marked with an ["incompatible-change"
label](https://github.com/bazelbuild/bazel/issues?q=label%3Aincompatible-change){: .external}.

For every incompatible change, the issue specifies the following:

*   Name of the flag controlling the incompatible change
*   Description of the changed functionality
*   Migration recipe

When an incompatible change is ready for migration with Bazel at HEAD
(therefore, also with the next Bazel rolling release), it should be marked with
the `migration-ready` label. The incompatible change issue is closed when the
incompatible flag is flipped at HEAD.