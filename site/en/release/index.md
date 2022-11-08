Project: /_project.yaml
Book: /_book.yaml

# Release Policy

{% dynamic setvar source_file "site/en/release/index.md" %}
{% include "_buttons.html" %}

Bazel maintains a
[Long Term Support (LTS)](/release/versioning)
release model, where a major version is released every nine months and minor
versions are released monthly. This page covers the Bazel release policy,
including the release candidates, timelines, announcements, and testing.

Bazel releases can be found on
[GitHub](https://github.com/bazelbuild/bazel/releases){: .external}.

## Release candidates {:#release-candidates}

A release candidate for a new version of Bazel is usually created at the
beginning of every month. The work is tracked by a
[release bug on GitHub](https://github.com/bazelbuild/bazel/issues?q=is%3Aissue+is%3Aopen+label%3Arelease){: .external}
indicating a target release date, and is assigned to the current Release manager.
Release candidates should pass all Bazel unit tests, and show no unwanted
regression in the projects tested on [Buildkite](https://buildkite.com/bazel){: .external}.

Release candidates are announced on
[bazel-discuss](https://groups.google.com/g/bazel-discuss){: .external}.
Over the next days, the Bazel team monitors community bug reports for any
regressions in the candidates.

## Releasing {:#releasing}

If no regressions are discovered, the candidate is officially released after
one week. However, regressions can delay the release of a release candidate. If
regressions are found, the Bazel team applies corresponding cherry-picks to the
release candidate to fix those regressions. If no further regressions are found
for two consecutive business days beginning after one week since the first
release candidate, the candidate is released.

New features are not cherry-picked into a release candidate after it is cut.
Moreover, if a new feature is buggy, the feature may be rolled back from a
release candidate. Only bugs that have the potential to highly impact or break
the release build are fixed in a release candidate after it is cut.

A release is only released on a day where the next day is a business day.

If a critical issue is found in the latest release, the Bazel team creates a
patch release by applying the fix to the release. Because this patch updates an
existing release instead of creating a new one, the patch release candidate can
be released after two business days.

## Testing {:#testing}

A nightly build of all projects running on
[ci.bazel.build](https://github.com/bazelbuild/continuous-integration/blob/master/buildkite/README.md){: .external} is run, using Bazel
binaries built at head, and release binaries. Projects going to be impacted by a
breaking change are notified.

When a release candidate is issued, other Google projects like
[TensorFlow](https://tensorflow.org){: .external} are tested on their complete
test suite using the release candidate binaries. If you have a critical project
using Bazel, we recommend that you establish an automated testing process that
tracks the current release candidate, and report any regressions.
