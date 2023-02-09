Project: /_project.yaml
Book: /_book.yaml

# Release Versioning

{% include "_buttons.html" %}

Bazel 4.0 and higher provides support for two release tracks: long term support
(LTS) releases and rolling releases. This page covers versioning in Bazel, the
types of releases, and the benefits of those releases for Bazel users and
contributors.

## Understanding versioning on Bazel {:#bazel-versioning}

Bazel uses a _major.minor.patch_ semantic versioning scheme.

* A _major release_ contains features that are not backward compatible with the
  previous release.
* A _minor release_ contains new backward-compatible features.
* A _patch release_ contains minor changes and bug fixes.

Using version 3.5.1 as an example, a new release of each type would result in
these version numbers:

* Major: 4.0
* Minor: 3.6
* Patch: 3.5.2

## Bazel's release cycle {:#release-cycle}

Bazel continually publishes rolling releases. Every major version is an LTS
release. You can choose to follow either release cadence - updating from one
LTS release to the next, or updating with each minor version release.

The image shows both rolling and LTS releases, and the expected support for
each.

![Roadmap](/docs/images/roadmap.png "Roadmap")

**Figure 1.** Rolling and LTS releases.

## Release branches {:#release-branches}

Each major version becomes a separate development branch on release. You can
receive fixes to critical bugs on that branch without having to update to the
Bazel release at head. Additional features on your major version branch become
minor releases and the highest version on the branch is the supported version.

Each Bazel release is paired with a list of recommended rule versions that work
together and there is strict backwards compatibility within each branch.

## LTS releases {:#lts-releases}

An LTS release is a major version (such as, 4.0) that is supported for 3 years
after its release.
A major version is released approximately every nine months.

Ongoing development on a release branch results in minor versions.

You can choose to pin your project to a major release and update to a newer
version in your own time. This gives you time to preview upcoming changes and
adapt to them in advance.

## Rolling releases {:#rolling-releases}

Rolling releases are periodically cut from Bazel's main branch.
This release cadence involves a continuous delivery of preview releases of the
next major Bazel version, which are in sync with Google’s internal Blaze
releases.

Note that a new rolling release can contain breaking changes that are
incompatible with previous releases.

Rolling releases are tested on Bazel's test suite on Bazel CI and
Google’s internal test suite. Incompatible flags may be
used to ease the burden of migrating to new functionality, but default behaviors
may change with any rolling release. (You can also use rolling releases to
preview the next LTS version. For example, `5.0.0-pre.20210604.6` is based on a
candidate cut on 2021-06-04 and represents a milestone towards the 5.0 LTS
release.)

You can download the latest rolling release from
[GitHub](https://github.com/bazelbuild/bazel/releases){: .external}.
Alternatively, you can set up
[Bazelisk v1.9.0](https://github.com/bazelbuild/bazelisk/releases/tag/v1.9.0){: .external}
(or later) to use a specific version name or the
“rolling” identifier, which uses the most recent rolling release. For more
details, see the
[Bazelisk documentation](https://github.com/bazelbuild/bazelisk#how-does-bazelisk-know-which-bazel-version-to-run){: .external}.

## Updating versions {:#updating-versions}

* For more information on updating your Bazel version, see
  [Updating Bazel](/install/bazelisk).
* For more information on contributing updates to new Bazel releases, see
  [Contributing to Bazel](/contribute).
