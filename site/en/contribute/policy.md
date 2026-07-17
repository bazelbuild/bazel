Project: /_project.yaml
Book: /_book.yaml
translation: human
page_type: lcat

# Contribution policy

{% include "_buttons.html" %}

This page covers Bazel's governance model and contribution policy.

## Governance model

The [Bazel project](https://github.com/bazelbuild){: .external} is led and managed by Google
and has a large community of contributors outside of Google. Some Bazel
components (such as specific rules repositories under the
[bazelbuild](https://github.com/bazelbuild){: .external} organization) are led,
maintained, and managed by members of the community. The Google Bazel team
reviews suggestions to add community-owned repositories (such as rules) to the
[bazelbuild](https://github.com/bazelbuild){: .external} GitHub organization.

### Contributor roles

Here are outlines of the roles in the Bazel project, including their
responsibilities:

*   **Owners**: The Google Bazel team. Owners are responsible for:
    *   Strategy, maintenance, and leadership of the Bazel project.
    *   Building and maintaining Bazel's core functionality.
    *   Appointing Maintainers and approving new repositories.
*   **Maintainers**: The Google Bazel team and designated GitHub users.
    Maintainers are responsible for:
    *   Building and maintaining the primary functionality of their repository.
    *   Reviewing and approving contributions to areas of the Bazel code base.
    *   Supporting users and contributors with timely and transparent issue
        management, PR review, and documentation.
    *   Releasing, testing and collaborating with Bazel Owners.
*   **Contributors**: All users who contribute code or documentation to the
    Bazel project.
    *   Creating well-written PRs to contribute to Bazel's codebase and
        documentation.
    *   Using standard channels, such as GitHub Issues, to propose changes and
        report issues.

### Becoming a Maintainer

Bazel Owners may appoint Maintainers to lead well-defined areas of code, such as
rule sets. Contributors with a record of consistent, responsible past
contributions who are planning major contributions in the future could be
considered to become qualified Maintainers.

## Contribution policy {:#contribution-policy}

The Bazel project accepts contributions from external contributors. Here are the
contribution policies for Google-managed and Community-managed areas of code.

*   **Licensing**. All Maintainers and Contributors must sign the
    [Google’s Contributor License Agreement](https://cla.developers.google.com/clas){: .external}.
*   **Contributions**. Owners and Maintainers should make every effort to accept
    worthwhile contributions. All contributions must be:
    *   Well written and well tested
    *   Discussed and approved by the Maintainers of the relevant area of code.
        Discussions and approvals happen on GitHub Issues and in GitHub PRs.
        Larger contributions require a
        [design review](/contribute/design-documents).
    *   Added to Bazel's Continuous Integration system if not already present.
    *   Supportable and aligned with Bazel product direction
*   **Code review**. All changes in all `bazelbuild` repositories require
    review:
    *   All PRs must be approved by an Owner or Maintainer.
    *   Only Owners and Maintainers can merge PRs.
*   **Compatibility**. Owners may need to reject or request modifications to PRs
    in the unlikely event that the change requires substantial modifications to
    internal Google systems.
*   **Documentation**. Where relevant, feature contributions should include
    documentation updates.

For more details on contributing to Bazel, see our
[contribution guidelines](/contribute/).
