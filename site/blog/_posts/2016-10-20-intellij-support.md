---
layout: posts
title: IntelliJ and Android Studio support
---

We've recently open-sourced Bazel plugins for
[IntelliJ and Android Studio](https://ij.bazel.io).

## Key Features ##

* Import a project directly from a BUILD file.
* BUILD file integration: syntax highlighting, refactoring, find usages,
  code completion, etc. [Skylark](/docs/skylark)
  extensions are fully supported.
* Compile your project and get navigatable Bazel compile errors in the IDE.
* [Buildifier](https://github.com/bazelbuild/buildifier) integration
* Support for Bazel run configurations for certain rule classes.
* Run/debug tests directly through Bazel by right-clicking on
  methods/classes/BUILD targets.

## How do I get started? ##

To try them out, you can install them directly from within the IDE
(**Settings > Plugins > Browse repositories**), download them from the
JetBrains [plugin repository](https://plugins.jetbrains.com/search/index?search=bazel),
or build directly from [source](https://github.com/bazelbuild/intellij).

Detailed docs are available [here](https://ij.bazel.io).

The plugins are currently Linux-only, with plans for Mac support in the future.
