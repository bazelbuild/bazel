---
layout: documentation
title: Updating Bazel
---

# Updating Bazel

The Bazel project has a [backwards compatibility
policy](https://docs.bazel.build/versions/master/backward-compatibility.html)
(see [guidance for rolling out incompatible
changes](https://www.bazel.build/breaking-changes-guide.html) if you are the
author of one). That page summarizes best practices on how to test and migrate
your project with upcoming incompatible changes and how to provide feedback to
the incompatible change authors.

## Managing Bazel versions with Bazelisk

The Bazel team implemented a Bazel wrapper called
[bazelisk](https://github.com/bazelbuild/bazelisk) that helps you manage Bazel
versions.

Bazelisk can:
*   Auto-update Bazel to the latest version
*   Build the project with a Bazel version specified in the .bazelversion
    file. Check in that file into your version control to ensure reproducibility
    of your builds.
*   Help migrate your project for incompatible changes (see above)
*   Easily try release candidates

## Recommended migration process

Bazel backwards compatibility policy is designed to avoid _upgrade cliffs_: any
project can be prepared for the next Bazel release without breaking
compatibility with the current release.

We recommend the following process for project migration:


1. Assume that your project already works with a given Bazel release, say 0.26,
   and you want to prepare for the next release, say 0.27
2. Find all incompatible changes for which the migration can be started: they are marked with
   "migration-\<release\>" label on GitHub, for example
   "[migration-0.26](https://github.com/bazelbuild/bazel/issues?utf8=%E2%9C%93&q=label%3Amigration-0.26+)".
3. Each of those issues has an associated `--incompatible_*` flag. For each of
   them, build your project with that flag enabled, and if the build is
   unsuccessful, fix the project according to [migration
   recipe](https://docs.bazel.build/versions/master/backward-compatibility.html#incompatible-changes-and-migration-recipes)
   as specified in the corresponding GitHub issue:
    *   Migration guidance is available in the associated GitHub issue.
    *   Migration is always possible in such a way that the project continues to build with and without the flag.
    *   For some of the incompatible changes migration tooling is available, for
        example as part of
        [buildifier](https://github.com/bazelbuild/buildtools/releases). Be sure
        to check the GitHub issue for migration instructions.
    *   Please report any migration problems by commenting associated GitHub issue.
4. After all changes are migrated, you can continue to build your project
   without any flags: it will be ready for the next Bazel release.


### Migrating with Bazelisk

[Bazelisk](https://github.com/bazelbuild/bazelisk) can
greatly simplify the migration process described above.

*   `bazelisk --strict` will build given targets with all incompatible flags for
     changes with appropriate migration-* labels.
*   `bazelisk --migrate` will do even more: it will try every flag and report
     those for which the build was unsuccessful
