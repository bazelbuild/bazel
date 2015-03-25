# Bazel Feature Roadmap

Author: [The Bazel Team](https://groups.google.com/forum/#!forum/bazel-discuss)

This document describes the Bazel team's plans for introducing future
system features. The document is divided into two main sections:

*   *Near-term plans:* Features that the team is committed to supporting
    for alpha and beta releases.
*   *Tentative post-beta plans:* Features that the team is considering
    supporting within one year of the beta release. The availability and
    sequencing of features on this list will depend technical feasibility
    and user demand.

Note that this roadmap only includes features that the Bazel team itself
intends to support. We anticipate that a number of other features will be
added by code contributors.

For the alpha and beta releases, the Bazel team will maintain two code
repositories:

*   A Google-internal repository, containing both Bazel code and
    Google-specific extensions and features
*   An external [GitHub repository](https://github.com/google/bazel),
    containing only the Bazel code.

We anticipate making the external repository *primary* in the future, that is,
code from Google and non-Google contributors will committed and tested in the
external repository first, then imported into the internal repository. For
the alpha and beta releases, however, the internal repository will be primary.
Changes to Bazel code will be frequently pushed from the internal to
the external repository.

## Near-term Plans

### <a name="alpha"></a>Alpha release (2015-03)

This release is intended to

*   give users access to a basic set of supported features
*   allow potential code contributors to understand Bazel's capabilities and
    architecture
*   enable users to monitor the team's progress toward a beta release

Alpha users will need to build Bazel from source; we will not provide binary
releases until the beta. We will accept code contributions, but may need to
reject or defer code contributions related to beta features still under
development. Some features will be fully supported
henceforward, others are still experimental or partially supported;
see our [feature support document](support.md) for details.

The following features/capabilities will be available in the alpha:

*   Bazel runs on Linux and OS X
*   Bazel builds itself on Linux and OS X
*   Build rules target executables running on Linux and OS X
*   Support for building and testing C++
*   Support for building and testing Java
*   Support for building Objective C
*   Support for building Python
*   Support for building iOS apps
*   Documentation for supported build rules and features
*   Online support for bug and feature requests
*   A basic Bazel test suite in GitHub
*   Support for referencing Java JAR dependencies from HTTP and Maven endpoints
*   Support for referencing remote source repositories via HTTP
*   Support for extensibility via an interpreted Python subset (Skylark)

### <a name="beta"></a>Beta release (target date: 2015-06)

The beta release will add support for additional languages and platform and
various other fully supported features. In particular, the following
features/capabilities will be available:

*   Binary versions of Bazel for Linux and OS X
*   All Bazel tests currently in Google's repository are ported to GitHub
*   The Bazel test suite runs on externally-visible continuous integration
    infrastructure
*   Support for referencing transitive sources via Maven
*   Support for prefetching and caching remote dependencies
*   Support for building and testing Android apps
*   Support for testing iOS apps
*   Support for build action sandboxing

## Tentative post-beta plans

*   Binary releases of Bazel at least monthly
*   The external repository is primary
*   An auto-update mechanism to allow Bazel to dynamically check for, fetch,
    and install updated releases, rules, etc.
*   AppEngine support
*   Support for remote mobile testing
*   Support for the Go language
*   Support for Javascript
*   Android Studio interoperability


