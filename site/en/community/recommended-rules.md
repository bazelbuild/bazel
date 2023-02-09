Project: /_project.yaml
Book: /_book.yaml

# Recommended Rules

{% include "_buttons.html" %}

In the documentation, we provide a list of
[recommended rules](/rules).

This is a set of high quality rules, which will provide a good experience to our
users. We make a distinction between the supported rules, and the hundreds of
rules you can find on the Internet.

## Nomination {:#nomination}

If a ruleset meets the requirements below, a rule maintainer can nominate it
to be part of the _recommended rules_ by filing a
[GitHub issue](https://github.com/bazelbuild/bazel/){: .external}.

After a review by the [Bazel core team](/contribute/policy), it
will be recommended on the Bazel website.

## Requirements for the rule maintainers {:#requirements-rule-maintainers}

*   The ruleset provides an important feature, useful to a large number of Bazel
    users (for example, support for a widely popular language).
*   The ruleset is well maintained. There must be at least two active maintainers.
*   The ruleset is well documented, with examples, and easy to use.
*   The ruleset follows the best practices and is performant (see
    [the performance guide](/rules/performance)).
*   The ruleset has sufficient test coverage.
*   The ruleset is tested on
    [BuildKite](https://github.com/bazelbuild/continuous-integration/blob/master/buildkite/README.md){: .external}
    with the latest version of Bazel. Tests should always pass (when used as a
    presubmit check).
*   The ruleset is also tested with the upcoming incompatible changes. Breakages
    should be fixed within two weeks. Migration issues should be reported to the
    Bazel team quickly.

## Requirements for Bazel developers {:#requirements-dev}

*   Recommended rules are frequently tested with Bazel at head (at least once a
    day).
*   No change in Bazel may break a recommended rule (with the default set of
    flags). If it happens, the change should be fixed or rolled back.

## Demotion {:#demotion}

If there is a concern that a particular ruleset is no longer meeting the
requirements, a [GitHub issue](https://github.com/bazelbuild/bazel/){: .external} should be
filed.

Rule maintainers will be contacted and need to respond in 2 weeks. Based on the
outcome, Bazel core team might make a decision to demote the rule set.
