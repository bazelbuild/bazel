Project: /_project.yaml
Book: /_book.yaml

# Rules

{% dynamic setvar source_file "site/en/rules/index.md" %}
{% include "_buttons.html" %}

The Bazel ecosystem has a growing and evolving set of rules to support popular
languages and packages. Much of Bazel's strength comes from the ability to
[define new rules](/extending/concepts) that can be used by others.

This page describes the recommended, native, and non-native Bazel rules.

## Recommended rules {:#recommended-rules}

Here is a selection of recommended rules:

* [Android](/docs/bazel-and-android)
* [C / C++](/docs/bazel-and-cpp)
* [Docker/OCI](https://github.com/bazel-contrib/rules_oci){: .external}
* [Go](https://github.com/bazelbuild/rules_go){: .external}
* [Haskell](https://github.com/tweag/rules_haskell){: .external}
* [Java](/docs/bazel-and-java)
* [JavaScript / NodeJS](https://github.com/bazelbuild/rules_nodejs){: .external}
* [Maven dependency management](https://github.com/bazelbuild/rules_jvm_external){: .external}
* [Objective-C](/docs/bazel-and-apple)
* [Package building](https://github.com/bazelbuild/rules_pkg){: .external}
* [Protocol Buffers](https://github.com/bazelbuild/rules_proto#protobuf-rules-for-bazel){: .external}
* [Python](https://github.com/bazelbuild/rules_python){: .external}
* [Scala](https://github.com/bazelbuild/rules_scala){: .external}
* [Shell](/reference/be/shell)
* [Webtesting](https://github.com/bazelbuild/rules_webtesting){: .external} (Webdriver)

The repository [Skylib](https://github.com/bazelbuild/bazel-skylib){: .external} contains
additional functions that can be useful when writing new rules and new
macros.

The rules above were reviewed and follow our
[requirements for recommended rules](/community/recommended-rules){: .external}.
Contact the respective rule set's maintainers regarding issues and feature
requests.

To find more Bazel rules, use a search engine, take a look on
[awesomebazel.com](https://awesomebazel.com/){: .external}, or search on
[GitHub](https://github.com/search?o=desc&q=bazel+rules&s=stars&type=Repositories){: .external}.

## Native rules that do not apply to a specific programming language

Native rules are shipped with the Bazel binary, they are always available in
BUILD files without a `load` statement.

* Extra actions
  - [`extra_action`](/reference/be/extra-actions#extra_action)
  - [`action_listener`](/reference/be/extra-actions#action_listener)
* General
  - [`filegroup`](/reference/be/general#filegroup)
  - [`genquery`](/reference/be/general#genquery)
  - [`test_suite`](/reference/be/general#test_suite)
  - [`alias`](/reference/be/general#alias)
  - [`config_setting`](/reference/be/general#config_setting)
  - [`genrule`](/reference/be/general#genrule)
* Platform
  - [`constraint_setting`](/reference/be/platforms-and-toolchains#constraint_setting)
  - [`constraint_value`](/reference/be/platforms-and-toolchains#constraint_value)
  - [`platform`](/reference/be/platforms-and-toolchains#platform)
  - [`toolchain`](/reference/be/platforms-and-toolchains#toolchain)
  - [`toolchain_type`](/reference/be/platforms-and-toolchains#toolchain_type)
* Workspace
  - [`bind`](/reference/be/workspace#bind)
  - [`local_repository`](/reference/be/workspace#local_repository)
  - [`new_local_repository`](/reference/be/workspace#new_local_repository)
  - [`xcode_config`](/reference/be/objective-c#xcode_config)
  - [`xcode_version`](/reference/be/objective-c#xcode_version)

## Embedded non-native rules {:#embedded-rules}

Bazel also embeds additional rules written in [Starlark](/rules/language). Those can be loaded from
the `@bazel_tools` built-in external repository.

* Repository rules
  - [`git_repository`](/rules/lib/repo/git#git_repository)
  - [`http_archive`](/rules/lib/repo/http#http_archive)
  - [`http_file`](/rules/lib/repo/http#http_archive)
  - [`http_jar`](/rules/lib/repo/http#http_jar)
  - [Utility functions on patching](/rules/lib/repo/utils)
