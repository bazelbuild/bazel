Project: /_project.yaml
Book: /_book.yaml

# Guide for rolling out breaking changes

{% include "_buttons.html" %}

It is inevitable that we will make breaking changes to Bazel. We will have to
change our designs and fix the things that do not quite work. However, we need
to make sure that community and Bazel ecosystem can follow along. To that end,
Bazel project has adopted a
[backward compatibility policy](/release/backward-compatibility).
This document describes the process for Bazel contributors to make a breaking
change in Bazel to adhere to this policy.

1. Follow the [design document policy](/contribute/design-documents).

1. [File a GitHub issue.](#github-issue)

1. [Implement the change.](#implementation)

1. [Update labels.](#labels)

1. [Update repositories.](#update-repos)

1. [Flip the incompatible flag.](#flip-flag)

## GitHub issue {:#github-issue}

[File a GitHub issue](https://github.com/bazelbuild/bazel/issues){: .external}
in the Bazel repository.
[See example.](https://github.com/bazelbuild/bazel/issues/6611){: .external}

We recommend that:

* The title starts with the name of the flag (the flag name will start with
  `incompatible_`).

* You add the label
  [`incompatible-change`](https://github.com/bazelbuild/bazel/labels/incompatible-change){: .external}.

* The description contains a description of the change and a link to relevant
  design documents.

* The description contains a migration recipe, to explain users how they should
  update their code. Ideally, when the change is mechanical, include a link to a
  migration tool.

* The description includes an example of the error message users will get if
  they don't migrate. This will make the GitHub issue more discoverable from
  search engines. Make sure that the error message is helpful and actionable.
  When possible, the error message should include the name of the incompatible
  flag.

For the migration tool, consider contributing to
[Buildifier](https://github.com/bazelbuild/buildtools/blob/master/buildifier/README.md){: .external}.
It is able to apply automated fixes to `BUILD`, `WORKSPACE`, and `.bzl` files.
It may also report warnings.

## Implementation {:#implementation}

Create a new flag in Bazel. The default value must be false. The help text
should contain the URL of the GitHub issue. As the flag name starts with
`incompatible_`, it needs metadata tags:

```java
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
      },
```

In the commit description, add a brief summary of the flag.
Also add [`RELNOTES:`](release-notes.md) in the following form:
`RELNOTES: --incompatible_name_of_flag has been added. See #xyz for details`

The commit should also update the relevant documentation, so that there is no
window of commits in which the code is inconsistent with the docs. Since our
documentation is versioned, changes to the docs will not be inadvertently
released prematurely.

## Labels {:#labels}

Once the commit is merged and the incompatible change is ready to be adopted, add the label
[`migration-ready`](https://github.com/bazelbuild/bazel/labels/migration-ready){: .external}
to the GitHub issue.

If a problem is found with the flag and users are not expected to migrate yet:
remove the flags `migration-ready`.

If you plan to flip the flag in the next major release, add label `breaking-change-X.0" to the issue.

## Updating repositories {:#update-repos}

Bazel CI tests a list of important projects at
[Bazel@HEAD + Downstream](https://buildkite.com/bazel/bazel-at-head-plus-downstream){: .external}. Most of them are often
dependencies of other Bazel projects, therefore it's important to migrate them to unblock the migration for the broader community. To monitor the migration status of those projects, you can use the [`bazelisk-plus-incompatible-flags` pipeline](https://buildkite.com/bazel/bazelisk-plus-incompatible-flags){: .external}.
Check how this pipeline works [here](https://github.com/bazelbuild/continuous-integration/tree/master/buildkite#checking-incompatible-changes-status-for-downstream-projects){: .external}.

Our dev support team monitors the [`migration-ready`](https://github.com/bazelbuild/bazel/labels/migration-ready){: .external} label. Once you add this label to the GitHub issue, they will handle the following:

1. Create a comment in the GitHub issue to track the list of failures and downstream projects that need to be migrated ([see example](https://github.com/bazelbuild/bazel/issues/17032#issuecomment-1353077469){: .external})

1. File Github issues to notify the owners of every downstream project broken by your incompatible change ([see example](https://github.com/bazelbuild/intellij/issues/4208){: .external})

1. Follow up to make sure all issues are addressed before the target release date

Migrating projects in the downstream pipeline is NOT entirely the responsibility of the incompatible change author, but you can do the following to accelerate the migration and make life easier for both Bazel users and the Bazel Green Team.

1. Send PRs to fix downstream projects.

1. Reach out to the Bazel community for help on migration (e.g. [Bazel Rules Authors SIG](https://bazel-contrib.github.io/SIG-rules-authors/)).

## Flipping the flag {:#flip-flag}

Before flipping the default value of the flag to true, please make sure that:

* Core repositories in the ecosystem are migrated.

    On the [`bazelisk-plus-incompatible-flags` pipeline](https://buildkite.com/bazel/bazelisk-plus-incompatible-flags){: .external},
    the flag should appear under `The following flags didn't break any passing Bazel team owned/co-owned projects`.

* All issues in the checklist are marked as fixed/closed.

* User concerns and questions have been resolved.

When the flag is ready to flip in Bazel, but blocked on internal migration at Google, please consider setting the flag value to false in the internal `blazerc` file to unblock the flag flip. By doing this, we can ensure Bazel users depend on the new behaviour by default as early as possible.

When changing the flag default to true, please:

* Use `RELNOTES[INC]` in the commit description, with the
    following format:
    `RELNOTES[INC]: --incompatible_name_of_flag is flipped to true. See #xyz for
    details`
    You can include additional information in the rest of the commit description.
* Use `Fixes #xyz` in the description, so that the GitHub issue gets closed
    when the commit is merged.
* Review and update documentation if needed.
* File a new issue `#abc` to track the removal of the flag.

## Removing the flag {:#remove-flag}

After the flag is flipped at HEAD, it should be removed from Bazel eventually.
When you plan to remove the incompatible flag:

* Consider leaving more time for users to migrate if it's a major incompatible change.
  Ideally, the flag  should be available in at least one major release.
* For the commit that removes the flag, use `Fixes #abc` in the description
  so that the GitHub issue gets closed when the commit is merged.
