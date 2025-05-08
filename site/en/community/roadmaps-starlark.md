Project: /_project.yaml
Book: /_book.yaml

# Starlark Roadmap

{% include "_buttons.html" %}

*Last verified: 2020-04-21*
([update history](https://github.com/bazelbuild/bazel-website/commits/master/roadmaps/starlark.md))

*Point of contact:* [laurentlb](https://github.com/laurentlb)

## Goal

Our goal is to make Bazel more extensible. Users should be able to easily
implement their own rules, and support new languages and tools. We want to
improve the experience of writing and maintaining those rules.

We focus on two areas:

* Make the language and API simple, yet powerful.
* Provide better tooling for reading, writing, updating, debugging, and testing the code.


## Q2 2020

Build health and Best practices:

* P0. Discourage macros without have a name, and ensure the name is a unique
  string literal. This work is focused on Google codebase, but may impact
  tooling available publicly.
* P0. Make Buildozer commands reliable with regard to selects and variables.
* P1. Make Buildifier remove duplicates in lists that we don’t sort because of
  comments.
* P1. Update Buildifier linter to recommend inlining trivial expressions.
* P2. Study use cases for native.existing_rule[s]() and propose alternatives.
* P2. Study use cases for the prelude file and propose alternatives.

Performance:

* P1. Optimize the Starlark interpreter using flat environments and bytecode
  compilation.

Technical debt reduction:

* P0. Add ability to port native symbols to Starlark underneath @bazel_tools.
* P1. Delete obsolete flags (some of them are still used at Google, so we need to
  clean the codebase first): `incompatible_always_check_depset_elements`,
  `incompatible_disable_deprecated_attr_params`,
  `incompatible_no_support_tools_in_action_inputs`, `incompatible_new_actions_api`.
* P1. Ensure the followin flags can be flipped in Bazel 4.0:
  `incompatible_disable_depset_items`, `incompatible_no_implicit_file_export`,
  `incompatible_run_shell_command_string`,
  `incompatible_restrict_string_escapes`.
* P1. Finish lib.syntax work (API cleanup, separation from Bazel).
* P2. Reduce by 50% the build+test latency of a trivial edit to Bazel’s Java packages.

Community:

* `rules_python` is active and well-maintained by the community.
* Continuous support for rules_jvm_external (no outstanding pull requests, issue
  triage, making releases).
* Maintain Bazel documentation infrastructure: centralize and canonicalize CSS
  styles across bazel-website, bazel-blog, docs
* Bazel docs: add CI tests for e2e doc site build to prevent regressions.

## Q1 2020

Build health and Best practices:

* Allow targets to track their macro call stack, for exporting via `bazel query`
* Implement `--incompatible_no_implicit_file_export`
* Remove the deprecated depset APIs (#5817, #10313, #9017).
* Add a cross file analyzer in Buildifier, implement a check for deprecated
  functions.

Performance:

* Make Bazel’s own Java-based tests 2x faster.
* Implement a Starlark CPU profiler.

Technical debt reduction:

* Remove 8 incompatible flags (after flipping them).
* Finish lib.syntax cleanup work (break dependencies).
* Starlark optimization: flat environment, bytecode compilation
* Delete all serialization from analysis phase, if possible
* Make a plan for simplifying/optimizing lib.packages

Community:

* Publish a Glossary containing definitions for all the Bazel-specific terms
