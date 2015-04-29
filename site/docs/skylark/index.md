# Skylark

#### A simple way to write custom build rules

Skylark is a work-in-progress project, which allows extending Bazel with new
rules or macros (composition of rules and macros).

## Goals

* **Power to the user**. We want to allow users to write new rules in a simple
  language without having to understand all of Bazel's internals. The core
  contributors have to write or review all changes to native Bazel rules (both
  inside and outside of Google). This is a lot of work and we have to push back
  on new rules.
* **Simplicity**. Skylark has been designed to be as simple as possible. Contrary
  to native rules, the code is very short and does not rely on complex
  inheritance.

* **Faster development**. With Skylark, rules are stored in the source tree. Modify one
  rule file and run Bazel to see the result immediately. Before Skylark, we
  had to rebuild and restart Bazel before seeing a change, this slowed down the
  development process a lot.

* **Faster release cycle**. Update one rule file, commit the changes to
  version control and everyone will use it when they sync. No need to wait for
  a native rule to be released with the next Bazel binary.

## Getting started

Read the [concepts](concepts.md) behind Skylark and try the
[cookbook examples](cookbook.md). To go further, read about the
[standard library](library.html).

