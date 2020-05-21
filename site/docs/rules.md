---
layout: documentation
title: Rules
---

# Rules

## Recommended rules

Much of Bazel's strength comes from the ability [to define new rules](skylark/concepts.html)
which can be used by others. There is a growing and evolving set of rules to
support popular languages and packages.

Here is a selection of recommended rules:

* [Android](bazel-and-android.html)
* [Boost](https://github.com/nelhage/rules_boost)
* [C / C++](bazel-and-cpp.html)
* [Docker](https://github.com/bazelbuild/rules_docker)
* [Go](https://github.com/bazelbuild/rules_go)
* [Haskell](https://github.com/tweag/rules_haskell)
* [Java](bazel-and-java.html)
* [JavaScript / NodeJS](https://github.com/bazelbuild/rules_nodejs)
* [Kubernetes](https://github.com/bazelbuild/rules_k8s)
* [Maven dependency management](https://github.com/bazelbuild/rules_jvm_external)
* [Objective C](bazel-and-apple.html)
* [Package building and fetching rules](https://github.com/bazelbuild/rules_pkg)
* [Protocol Buffers](https://github.com/bazelbuild/rules_proto#protobuf-rules-for-bazel)
* [Python](https://github.com/bazelbuild/rules_python)
* [Scala](https://github.com/bazelbuild/rules_scala)
* [Shell](be/shell.html)
* [Webtesting](https://github.com/bazelbuild/rules_webtesting) (Webdriver)

The repository [Skylib](https://github.com/bazelbuild/bazel-skylib) contains
additional functions that can be useful when writing new rules and new
macros.

The rules above were reviewed and follow our
[requirements for recommended rules](https://www.bazel.build/recommended-rules.html).
Contact the respective rule set's maintainers regarding issues and feature
requests.

To find more Bazel rules, use a search engine, take a look on
[awesomebazel.com](https://awesomebazel.com/), or search on
[GitHub](https://github.com/search?o=desc&q=bazel+rules&s=stars&type=Repositories).

## Native rules that do not apply to a specific programming language

Native rules are shipped with the Bazel binary, they are always available in
BUILD files without a `load` statement.

* Extra actions
  - [`extra_action`](be/extra-actions.html#extra_action)
  - [`action_listener`](be/extra-actions.html#action_listener)
* General
  - [`filegroup`](be/general.html#filegroup)
  - [`genquery`](be/general.html#genquery)
  - [`test_suite`](be/general.html#test_suite)
  - [`alias`](be/general.html#alias)
  - [`config_setting`](be/general.html#config_setting)
  - [`genrule`](be/general.html#genrule)
* Platform
  - [`constraint_setting`](be/platform.html#constraint_setting)
  - [`constraint_value`](be/platform.html#constraint_value)
  - [`platform`](be/platform.html#platform)
  - [`toolchain`](be/platform.html#toolchain)
  - [`toolchain_type`](be/platform.html#toolchain_type)
* Workspace
  - [`bind`](be/workspace.html#bind)
  - [`local_repository`](be/workspace.html#local_repository)
  - [`new_local_repository`](be/workspace.html#new_local_repository)
  - [`xcode_config`](be/workspace.html#xcode_config)
  - [`xcode_version`](be/workspace.html#xcode_version)

## Embedded non-native rules

Bazel also embeds additional rules written in Starlark. Those can be loaded from
the `@bazel_tools` built-in external repository.

* Repository rules
  - [`git_repository`](repo/git.md#git_repository)
  - [`new_git_repository`](repo/git.html#new_git_repository)
  - [`http_archive`](repo/http.html#http_archive)
  - [`http_file`](repo/http.html#http_archive)
  - [`http_jar`](repo/http.html#http_jar)
  - [Utility functions on patching](repo/utils.md)
