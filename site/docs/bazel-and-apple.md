---
layout: documentation
title: Apple apps and Bazel
---

# Apple apps and Bazel

This page contains resources that help you use Bazel to build macOS and iOS
projects. It links to a tutorial, build rules, and other information specific to
using Bazel to build and test for those platforms.

## Contents

- [Working with Bazel](#working-with-bazel)
- [Migrating to Bazel](#migrating-to-bazel)
- [Apple apps and Skylark](#apple-apps-and-skylark)

## Working with Bazel

The following resources will help you work with Bazel on macOS and iOS projects:

*  [Tutorial: Building an iOS app](tutorial/ios-app.html)
*  [Objective-C build rules](https://docs.bazel.build/versions/master/be/objective-c.html)
*  [General Apple rules](https://github.com/bazelbuild/rules_apple)
*  [Integration with Xcode](ide.html)

## Migrating to Bazel

If you currently build your macOS and iOS projects with Xcode, follow the steps
in the migration guide to start building them with Bazel:

*  [Migrating from Xcode to Bazel](migrate-xcode.html)

## Apple apps and Skylark

**Note**: Extending Bazel with Skylark is for advanced build and test scenarios.
You do not need to use Skylark when getting started with Bazel.

The following [Skylark](https://docs.bazel.build/versions/master/skylark/concepts.html)
modules, configuration fragments, and providers will help you extend Bazel's
capabilities when building your macOS and iOS projects:

*  Modules:

   *  [`apple_bitcode_mode`](skylark/lib/apple_bitcode_mode.html)
   *  [`apple_common`](skylark/lib/apple_common.html)
   *  [`apple_platform`](skylark/lib/apple_platform.html)
   *  [`apple_platform_type`](skylark/lib/apple_platform_type.html)
   *  [`apple_toolchain`](skylark/lib/apple_toolchain.html)
   *  [`XcodeVersionConfig`](skylark/lib/XcodeVersionConfig.html)

*  Configuration fragments:

   *  [`apple`](skylark/lib/apple.html)

*  Providers:

   *  [`ObjcProvider`](skylark/lib/ObjcProvider.html)
   *  [`XcTestAppProvider`](skylark/lib/XcTestAppProvider.html)
