---
layout: documentation
title: IDE integration
---

# Integrating Bazel with IDEs

IDEs integrate with Bazel in a variety of ways, from features that allow Bazel executions from within the IDE, to awareness of Bazel structures such as syntax highlighting of the BUILD files.

## Supported IDEs and editors

### IntelliJ, Android Studio and CLion

*Status*: Beta, supported by the Bazel team.

[Plug-ins](http://ij.bazel.build) for Android Studio, CLion, and IntelliJ enable you to:

*  Import BUILD files into the IDE
*  Make your IDE aware of Starlark, the language used for Bazel's BUILD and bzl files
*  Build, test, and execute binaries directly from the IDE

Installation:

*  [Android Studio plugin](https://plugins.jetbrains.com/plugin/9185-android-studio-with-bazel)
*  [CLion plugin](https://plugins.jetbrains.com/plugin/9554-clion-with-bazel)
*  [IntelliJ plugin](https://plugins.jetbrains.com/plugin/8609-intellij-with-bazel)

### Xcode

*Status*: Beta, supported by the Bazel team.

[Tulsi](http://tulsi.bazel.build) generates Bazel-compatible Xcode projects from Bazel's `BUILD` files.

### Eclipse

*Status*: experimental, not officially supported by the Bazel team.

See [installation steps on GitHub](https://github.com/bazelbuild/eclipse#installation)


### Visual Studio Code

*Status*: Officially supported by the Bazel team.

Features:

* Bazel Build Targets tree
* Starlark debugger for .bzl files during a build (set breakpoints, step through code, inspect variables, etc.)

See [vscode-bazel in Visual Studio marketplace](https://marketplace.visualstudio.com/items?itemName=BazelBuild.vscode-bazel)

There is also an unofficial plugin that provides syntax highlighting and
formatting for Bazel `BUILD` and `WORKSPACE` files: [bazel-code](https://marketplace.visualstudio.com/items?itemName=DevonDCarew.bazel-code)


### Atom

*Status*: not officially supported by the Bazel team.

See [`language-bazel` package](https://atom.io/packages/language-bazel)



### Vim

*Status*: Beta, not officially supported by the Bazel team.

See [`bazelbuild/vim-bazel`on GitHub](https://github.com/bazelbuild/vim-bazel)

## Automatically run build and test on file change

[Bazel watcher](https://github.com/bazelbuild/bazel-watcher) is a tool for building Bazel targets when source files change.

## Building your own IDE plugin

Read the [*IDE support* blog post](https://bazel.build/blog/2016/06/10/ide-support.html) to learn more about the Bazel APIs to use when building an IDE plugin.
