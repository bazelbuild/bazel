Project: /_project.yaml
Book: /_book.yaml

# Guide for rolling out breaking changes

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

1. [Update labels](#labels)

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

* The description includes the intended length of migration window.

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
`RELNOTES: --incompatible_name_of_flag has been added. See #yxz for details`

The commit should also update the relevant documentation. As the documentation
is versioned, we recommend updating the documentation in the same commit
as the code change.


## Labels {:#labels}

Once the commit is merged, add the label
[`migration-ready`](https://github.com/bazelbuild/bazel/labels/migration-ready){: .external}
to the GitHub issue.

Later a [Bazel release manager](https://github.com/bazelbuild/continuous-integration/blob/master/docs/release-playbook.md){: .external}
will update the issue and replace the label with `migration-xx.yy`.

The label `breaking-change-xx.yy` communicates when we plan to flip the flag. If
a migration window needs to be extended, the author updates the label on GitHub
issue accordingly.

If a problem is found with the flag and users are not expected to migrate yet:
remove the flags `migration-xx.yy`.

## Updating repositories {:#update-repos}

1. Ensure that the core repositories are migrated. On the
  [`bazelisk-plus-incompatible-flags` pipeline](https://buildkite.com/bazel/bazelisk-plus-incompatible-flags){: .external},
  the flag should appear under "The following flags didn't break any passing Bazel team owned/co-owned projects".
1. Notify the owners of the other repositories.
1. Wait 14 days after the notifications to proceed, or until the flag is under
   "The following flags didn't break any passing projects" in the CI.

## Flipping the flag {:#flip-flag}

Before flipping the default value of the flag to true, please make sure that:

  * The migration window is respected.
  * User concerns and questions have been resolved.

When changing the flag default to true, please:

  * Use `RELNOTES[INC]` in the commit description, with the
    following format:
    `RELNOTES[INC]: --incompatible_name_of_flag is flipped to true. See #yxz for
    details`
    You can include additional information in the rest of the commit description.
  * Use `Fixes #xyz` in the description, so that the GitHub issue gets closed
    when the commit is merged.
  * Review and update documentation if needed.
