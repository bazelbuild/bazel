---
layout: documentation
title: Updating Bazel
category: getting-started
---

<div style="background-color: #EFCBCB; color: #AE2B2B;  border: 1px solid #AE2B2B; border-radius: 5px; border-left: 10px solid #AE2B2B; padding: 0.5em;">
<b>IMPORTANT:</b> The Bazel docs have moved! Please update your bookmark to <a href="https://bazel.build/versions/updating-bazel" style="color: #0000EE;">https://bazel.build/versions/updating-bazel</a>
<p/>
You can <a href="https://blog.bazel.build/2022/02/17/Launching-new-Bazel-site.html" style="color: #0000EE;">read about</a> the migration, and let us <a href="https://forms.gle/onkAkr2ZwBmcbWXj7" style="color: #0000EE;">know what you think</a>.
</div>


# Updating Bazel

This page covers how to automatically update your Bazel version using Bazelisk.

The Bazel project has a [backward compatibility
policy](https://docs.bazel.build/versions/main/backward-compatibility.html)
(see [guidance for rolling out incompatible
changes](https://www.bazel.build/maintaining/breaking-changes-guide.html) if you
are the author of one). That page summarizes best practices on how to test and
migrate your project with upcoming incompatible changes and how to provide
feedback to the incompatible change authors.

## Managing Bazel versions with Bazelisk

[Bazelisk](https://github.com/bazelbuild/bazelisk) helps you manage Bazel
versions.

Bazelisk can:
*   Auto-update Bazel to the latest LTS or rolling release.
*   Build the project with a Bazel version specified in the .bazelversion
    file. Check in that file into your version control to ensure reproducibility
    of your builds.
*   Help migrate your project for incompatible changes (see above)
*   Easily try release candidates

## Recommended migration process

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
        example, [buildifier](https://github.com/bazelbuild/buildtools/releases).
    *   Report migration problems by commenting on the associated GitHub issue.

After migration, you can continue to build your projects without worrying about
backward-compatibility until the next major release.
