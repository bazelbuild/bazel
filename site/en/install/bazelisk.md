Project: /_project.yaml
Book: /_book.yaml

# Installing / Updating Bazel using Bazelisk

{% include "_buttons.html" %}

## Installing Bazel

[Bazelisk](https://github.com/bazelbuild/bazelisk){: .external} is the
recommended way to install Bazel on Ubuntu, Windows, and macOS. It automatically
downloads and installs the appropriate version of Bazel. Use Bazelisk if you
need to switch between different versions of Bazel depending on the current
working directory, or to always keep Bazel updated to the latest release.

For more details, see
[the official README](https://github.com/bazelbuild/bazelisk/blob/master/README.md){: .external}.

## Updating Bazel

Bazel has a [backward compatibility policy](/release/backward-compatibility)
(see [guidance for rolling out incompatible
changes](/contribute/breaking-changes) if you
are the author of one). That page summarizes best practices on how to test and
migrate your project with upcoming incompatible changes and how to provide
feedback to the incompatible change authors.

### Managing Bazel versions with Bazelisk {:#manage-with-bazelisk}

[Bazelisk](https://github.com/bazelbuild/bazelisk){: .external} helps you manage
Bazel versions.

Bazelisk can:

*   Auto-update Bazel to the latest LTS or rolling release.
*   Build the project with a Bazel version specified in the .bazelversion
    file. Check in that file into your version control to ensure reproducibility
    of your builds.
*   Help migrate your project for incompatible changes (see above)
*   Easily try release candidates

### Recommended migration process {:#migration-process}

Within minor updates to any LTS release, any
project can be prepared for the next release without breaking
compatibility with the current release. However, there may be
backward-incompatible changes between major LTS versions.

Follow this process to migrate from one major version to another:

1. Read the release notes to get advice on how to migrate to the next version.
1. Major incompatible changes should have an associated `--incompatible_*` flag
   and a corresponding GitHub issue:
    *   Migration guidance is available in the associated GitHub issue.
    *   Tooling is available for some of incompatible changes migration. For
        example, [buildifier](https://github.com/bazelbuild/buildtools/releases){: .external}.
    *   Report migration problems by commenting on the associated GitHub issue.

After migration, you can continue to build your projects without worrying about
backward-compatibility until the next major release.
