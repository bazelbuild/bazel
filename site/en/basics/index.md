Project: /_project.yaml
Book: /_book.yaml

# Build Basics

{% dynamic setvar source_file "site/en/basics/index.md" %}
{% include "_buttons.html" %}

A build system is one of the most important parts of an engineering organization
because each developer interacts with it potentially dozens or hundreds of times
per day. A fully featured build system is necessary to enable developer
productivity as an organization scales. For individual developers, it's
straightforward to just compile your code and so a build system might seem
excessive. But at a larger scale, having a build system helps with managing
shared dependencies, such as relying on another part of the code base, or an
external resource, such as a library. Build systems help to make sure that you
have everything you need to build your code before it starts building. Build
systems also increase velocity when they're set up to help engineers share
resources and results.

This section covers some history and basics of building and build systems,
including design decisions that went into making Bazel. If you're
familiar with artifact-based build systems, such as Bazel, Buck, and Pants, you
can skip this section, but it's a helpful overview to understand why
artifact-based build systems are excellent at enabling scale.

Note: Much of this section's content comes from the _Build Systems and
Build Philosophy_ chapter of the
[_Software Engineering at Google_ book](https://abseil.io/resources/swe-book/html/ch18.html).
Thank you to the original author, Erik Kuefler, for allowing its reuse and
modification here!

*   **[Why a Build System?](/basics/build-systems)**

    If you haven't used a build system before, start here. This page covers why
    you should use a build system, and why compilers and build scripts aren't
    the best choice once your organization starts to scale beyond a few
    developers.

*   **[Task-Based Build Systems](/basics/task-based-builds)**

    This page discusses task-based build systems (such as Make, Maven, and
    Gradle) and some of their challenges.

*   **[Artifact-Based Build Systems](/basics/artifact-based-builds)**

    This page discusses artifact-based build systems in response to the pain
    points of task-based build systems.

*   **[Distributed Builds](/basics/distributed-builds)**

    This page covers distributed builds, or builds that are executed outside of
    your local machine. This requires more robust infrastructure to share
    resources and build results (and is where the true wizardry happens!)

*   **[Dependency Management](/basics/dependencies)**

    This page covers some complications of dependencies at a large scale and
    strategies to counteract those complications.
