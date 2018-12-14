---
layout: documentation
title: Bazel Vision
---

# Bazel Vision

<font size='+1'>Any software developer can efficiently build, test, and package
any project, of any size or complexity, with tooling that's easy to adopt and
extend.</font>

*   **Engineers can take build fundamentals for granted.** Software developers
    focus on the creative process of authoring code because the mechanical
    process of build and test is solved. When customizing the build system to
    support new languages or unique organizational needs, users focus on the
    aspects of extensibility that are unique to their use case, without having
    to reinvent the basic plumbing.
*   **Engineers can easily contribute to any project.** A developer who wants to
    start working on a new project can simply clone the project and run the
    build. There’s no need for local configuration - it just works. With
    cross-platform remote execution, they can work on any machine anywhere and
    fully test their changes against all platforms the project targets.
    Engineers can quickly configure the build for a new project or incrementally
    migrate an existing build.
*   **Projects can scale to any size codebase, any size team.** Fast,
    incremental testing allows teams to fully validate every change before it is
    committed. This remains true even as repos grow, projects span multiple
    repos, and multiple languages are introduced. Infrastructure does not force
    developers to trade test coverage for build speed.

**We believe Bazel has the potential to fulfill this vision.** Bazel was built
from the ground up to enable builds that are reproducible (a given set of inputs
will always produce the same outputs) and portable (a build can be run on any
machine without affecting the output). These characteristics support safe
incrementality (rebuilding only changed inputs doesn't introduce the risk of
corruption) and distributability (build actions are isolated and can be
offloaded). By minimizing the work needed to do a correct build and
parallelizing that work across multiple cores and remote systems, Bazel can make
any build fast. Bazel’s abstraction layer—instructions specific to languages,
platforms, and toolchains implemented in a simple extensibility language —
allows it to be easily applied to any context.

## Bazel Core Competencies

1.  Bazel supports **multi-language, multi-platform** builds and tests. You can
    run a single command to build and test your entire source tree, no matter
    which combination of languages and platforms you target.
1.  Bazel builds are **fast and correct**. Every build and test run is
    incremental, on your developers' machines and on CI.
1.  Bazel provides a **uniform, extensible language** to define builds for any
    language or platform.
1.  Bazel allows your builds **to scale** by connecting to remote execution and
    caching services.
1.  Bazel works across **all major development platforms** (Linux, MacOS, and
    Windows).
1.  We accept that adopting Bazel requires effort, but **gradual adoption** is
    possible. Bazel interfaces with de-facto standard tools for a given
    language/platform.

## Serving language communities

Software engineering evolves in the context of language communities — typically,
self-organizing groups of people who use common tools and practices. To be of
use to members of a language community, high-quality Bazel rules must be
available that integrate with the workflows and conventions of that community.
Bazel is committed to be extensible and open, and to support good rulesets for
any language.

### So what is a good ruleset?

1.  The rules need to support efficient **building and testing** for the
    language, including code coverage.
1.  The rules need to **interface with a widely-used "package manager"** for the
    language (such as Maven for Java), and support incremental migration paths
    from other widely-used build systems.
1.  The rules need to be **extensible and interoperable**, following
    ["Bazel sandwich"](https://bazel.build/designs/2016/08/04/extensibility-for-native-rules.html)
    principles.
1.  The rules need to be **remote-execution ready**. In practice, this means
    **configurable using the
    [toolchains](master/toolchains.html)
    mechanism**.
1.  The rules (and Bazel) need to interface with a **widely-used IDE** for the
    language, if there is one.
1.  The rules need to have **thorough, usable documentation,** with introductory
    material for new users, comprehensive docs for expert users.

Each of these items is essential and only together do they deliver on Bazel's
competencies for their particular ecosystem. They are also, by and large,
sufficient - once all are fulfilled, Bazel fully delivers its value to members
of that language community.
