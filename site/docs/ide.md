---
layout: documentation
title: IDE integration
---

# Integrating Bazel with IDEs

IDEs integrate with Bazel in a variety of ways, from features that allow Bazel
executions from within the IDE, to awareness of Bazel structures such as syntax
highlighting of the BUILD files.

If you are interested in developing an editor or IDE plugin for Bazel, please
join the `#ide` channel on the [Bazel Slack](https://slack.bazel.build) or email
[bazel-dev@googlegroups.com](bazel-dev@googlegroups.com).

## IDEs and editors

### IntelliJ, Android Studio and CLion

[Official plugin](http://ij.bazel.build) for IntelliJ, Android Studio, and
CLion. The plugin is [open source](https://github.com/bazelbuild/intellij).

This is the open source version of the plugin used internally at Google.

Features:

* Interop with language-specific plugins. Supported languages include Java,
  Scala, and Python.
* Import BUILD files into the IDE with semantic awareness of Bazel targets.
* Make your IDE aware of Starlark, the language used for Bazel's BUILD and bzl
  files
* Build, test, and execute binaries directly from the IDE
* Create configurations for debugging and running binaries.

To install, go to the IDE's plugin browser and search for `Bazel`.

To manually install older versions, download the zip files from JetBrains'
Plugin Repository and install the zip file from the IDE's plugin browser:

*  [Android Studio
   plugin](https://plugins.jetbrains.com/plugin/9185-android-studio-with-bazel)
*  [IntelliJ
   plugin](https://plugins.jetbrains.com/plugin/8609-intellij-with-bazel)
*  [CLion plugin](https://plugins.jetbrains.com/plugin/9554-clion-with-bazel)

### Xcode

[Tulsi](https://tulsi.bazel.build) and
[XCHammer](https://github.com/pinterest/xchammer) generate Xcode projects from
Bazel `BUILD` files.

### Visual Studio Code

Official plugin for VS Code.

Features:

* Bazel Build Targets tree
* Starlark debugger for .bzl files during a build (set breakpoints, step through
  code, inspect variables, and so on)

Find [the plugin on the Visual Studio
marketplace](https://marketplace.visualstudio.com/items?itemName=BazelBuild.vscode-bazel).
The plugin is [open source](https://github.com/bazelbuild/vscode-bazel).


### Atom

Find the [`language-bazel` package](https://atom.io/packages/language-bazel) on
the Atom package manager.

### Vim

See [`bazelbuild/vim-bazel` on GitHub](https://github.com/bazelbuild/vim-bazel)

### Emacs

See [`bazelbuild/bazel-emacs-mode` on
GitHub](https://github.com/bazelbuild/emacs-bazel-mode)

### Visual Studio

[Lavender](https://github.com/tmandry/lavender) is an experimental project for
generating Visual Studio projects that use Bazel for building.

## Automatically run build and test on file change

[Bazel watcher](https://github.com/bazelbuild/bazel-watcher) is a tool for
building Bazel targets when source files change.

## Building your own IDE plugin

Read the [**IDE support** blog
post](https://bazel.build/blog/2016/06/10/ide-support.html) to learn more about
the Bazel APIs to use when building an IDE plugin.

## Archived projects

These projects are no longer supported by the Bazel team.

* [Eclipse](https://github.com/bazelbuild/eclipse)
