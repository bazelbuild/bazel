---
layout: documentation
title: Rules
---

# Rules

Much of Bazel's strength comes from the ability [to define new rules](skylark/concepts.html)
which can be used by others. There is a growing and evolving set of rules to
support popular languages and packages.

## Recommended rules

* [Android](bazel-and-android.html)
* [C / C++](bazel-and-cpp.html)
* [Java](bazel-and-java.html)
* [Objective C](bazel-and-apple.html)
* [Protocol Buffers](https://github.com/bazelbuild/rules_proto#protobuf-rules-for-bazel)
* [Python](https://github.com/bazelbuild/rules_python)
* [Shell](be/shell.html)

## Additional rules

Rules for many popular languages have been created and are being maintained
outside of Bazel. Contact the respective rule set's maintainers regarding issues
and feature requests.

To find more Bazel rules, use a search engine or look on
[GitHub](https://github.com/search?o=desc&q=bazel+rules&s=stars&type=Repositories).

Here is a selection of popular rules:

* [Android](https://github.com/bazelbuild/rules_android)
* [Apple platforms](https://github.com/bazelbuild/rules_apple)
* [Boost](https://github.com/nelhage/rules_boost)
* [C#](https://github.com/bazelbuild/rules_dotnet)
* [Closure](https://github.com/bazelbuild/rules_closure)
* [Docker](https://github.com/bazelbuild/rules_docker)
* [Go](https://github.com/bazelbuild/rules_go)
* [Haskell](https://github.com/tweag/rules_haskell)
* [Jsonnet](https://github.com/bazelbuild/rules_jsonnet)
* [Kotlin](https://github.com/bazelbuild/rules_kotlin)
* [Kubernetes](https://github.com/bazelbuild/rules_k8s)
* [NodeJS](https://github.com/bazelbuild/rules_nodejs)
* [Protobuf](https://github.com/pubref/rules_protobuf)
* [Python](https://github.com/bazelbuild/rules_python)
* [Rust](https://github.com/bazelbuild/rules_rust)
* [Scala](https://github.com/bazelbuild/rules_scala)
* [Swift](https://github.com/bazelbuild/rules_swift)
* [Typescript](https://github.com/bazelbuild/rules_typescript)
* [Webtesting](https://github.com/bazelbuild/rules_webtesting) (Webdriver)

The repository [Skylib](https://github.com/bazelbuild/bazel-skylib) contains
additional functions that can be useful when writing new rules and new
macros.

## Native rules that do not apply to a specific programming language

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
* Workspace
  - [`bind`](be/workspace.html#bind)
  - [`local_repository`](be/workspace.html#local_repository)
  - [`maven_jar`](be/workspace.html#maven_jar)
  - [`maven_server`](be/workspace.html#maven_server)
  - [`new_local_repository`](be/workspace.html#new_local_repository)
  - [`xcode_config`](be/workspace.html#xcode_config)
  - [`xcode_version`](be/workspace.html#xcode_version)

## Embedded non-native rules

Bazel also embeds additional rules written in Starlark. Those can be loaded from
the `@bazel_tools` built-in external repository.

* Repository rules
  - [`git_repository`](repo/git.md#git_repository),
    [`new_git_repository`](repo/git.html#new_git_repository)
  - [`http_archive`](repo/http.html#http_archive),
    [`http_file`](repo/http.html#http_archive),
    [`http_jar`](repo/http.html#http_jar)
  - [Utility functions on patching](utils.md)

* Package rules: [`pkg_tar`](be/pkg.html#), [`pkg_deb`](be/pkg.html#pkg_deb), [`pkg_rpm`](be/pkg.html#pkg_rpm)
