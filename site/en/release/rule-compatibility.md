Project: /_project.yaml
Book: /_book.yaml

# Rule Compatibility

{% include "_buttons.html" %}

Bazel Starlark rules can break compatibility with Bazel LTS releases in the
following two scenarios:

1.  The rule breaks compatibility with future LTS releases because a feature it
    depends on is removed from Bazel at HEAD.
1.  The rule breaks compatibility with the current or older LTS releases because
    a feature it depends on is only available in newer Bazel LTS releases.

Meanwhile, the rule itself can ship incompatible changes for their users as
well. When combined with breaking changes in Bazel, upgrading the rule version
and Bazel version can often be a source of frustration for Bazel users. This
page covers how rules authors should maintain rule compatibility with Bazel to
make it easier for users to upgrade Bazel and rules.

## Manageable migration process {:#manageable-migration-process}

While it's obviously not feasible to guarantee compatibility between every
version of Bazel and every version of the rule, our aim is to ensure that the
migration process remains manageable for Bazel users. A manageable migration
process is defined as a process where **users are not forced to upgrade the
rule's major version and Bazel's major version simultaneously**, thereby
allowing users to handle incompatible changes from one source at a time.

For example, with the following compatibility matrix:

*   Migrating from rules_foo 1.x + Bazel 4.x to rules_foo 2.x + Bazel 5.x is not
    considered manageable, as the users need to upgrade the major version of
    rules_foo and Bazel at the same time.
*   Migrating from rules_foo 2.x + Bazel 5.x to rules_foo 3.x + Bazel 6.x is
    considered manageable, as the users can first upgrade rules_foo from 2.x to
    3.x without changing the major Bazel version, then upgrade Bazel from 5.x to
    6.x.

| | rules_foo 1.x | rules_foo 2.x | rules_foo 3.x | HEAD |
| --- | --- | --- | --- | --- |
| Bazel 4.x | ✅ | ❌ | ❌ | ❌ |
| Bazel 5.x | ❌ | ✅ | ✅ | ❌ |
| Bazel 6.x | ❌ | ❌ | ✅ | ✅ |
| HEAD | ❌ | ❌ | ❌ | ✅ |

❌: No version of the major rule version is compatible with the Bazel LTS
release.

✅: At least one version of the rule is compatible with the latest version of the
Bazel LTS release.

## Best practices {:#best-practices}

As Bazel rules authors, you can ensure a manageable migration process for users
by following these best practices:

1.  The rule should follow [Semantic
    Versioning](https://semver.org/){: .external}: minor versions of the same
    major version are backward compatible.
1.  The rule at HEAD should be compatible with the latest Bazel LTS release.
1.  The rule at HEAD should be compatible with Bazel at HEAD. To achieve this,
    you can
    *   Set up your own CI testing with Bazel at HEAD
    *   Add your project to [Bazel downstream
        testing](https://github.com/bazelbuild/continuous-integration/blob/master/docs/downstream-testing.md){: .external};
        the Bazel team files issues to your project if breaking changes in Bazel
        affect your project, and you must follow our [downstream project
        policies](https://github.com/bazelbuild/continuous-integration/blob/master/docs/downstream-testing.md#downstream-project-policies){: .external}
        to address issues timely.
1.  The latest major version of the rule must be compatible with the latest
    Bazel LTS release.
1.  A new major version of the rule should be compatible with the last Bazel LTS
    release supported by the previous major version of the rule.

Achieving 2. and 3. is the most important task since it allows achieving 4. and
5.  naturally.

To make it easier to keep compatibility with both Bazel at HEAD and the latest
Bazel LTS release, rules authors can:

*   Request backward-compatible features to be back-ported to the latest LTS
    release, check out [release process](/release#release-procedure-policies)
    for more details.
*   Use [bazel_features](https://github.com/bazel-contrib/bazel_features){: .external}
    to do Bazel feature detection.

In general, with the recommended approaches, rules should be able to migrate for
Bazel incompatible changes and make use of new Bazel features at HEAD without
dropping compatibility with the latest Bazel LTS release.