---
layout: documentation
title: Converting CocoaPods dependencies
---

# Converting CocoaPods dependencies

This document provides high-level guidelines for converting [CocoaPods](https://www.cocoapods.org/)
dependencies to Bazel packages that are compatible with [Tulsi](https://tulsi.bazel.build/).
CocoaPods is a third-party dependency management system for Apple application
development.

**Note:**  CocoaPods conversion is a manual process with many variables.
CocoaPods integration with Bazel has not been fully verified and is not
officially supported.

## Analyze your CocoaPods dependencies

If you're using CocoaPods, you need to:

1.  Examine the `Podfile` files to determine the hierarchy of the `Podspecs`.

2.  Take note of the version numbers in the corresponding `Podfile.lock` files
    to ensure that you are pulling the correct `Podspecs`.

3.  Document the dependency tree, including the hierarchy of the `Podspecs`,
    resource URLs, filenames, and version numbers.

## Converting a `Podspec` to a Bazel package

To convert a `Podspec` dependency to a Bazel package, do the following:

1. Download each `Podspec` and decompress it into its own directory within the
   Bazel workspace. All `Podspec`s must reside within the same Bazel workspace
   for Tulsi to be aware of them for inclusion in the Xcode project.

2. Within the `Podspec` directory, create a `BUILD` file that specifies the
   library target(s) referencing the source and header files on which your
   project depends.

3. Based on your project's dependency tree, add the `Podspec`s target(s) as
   dependencies to the appropriate targets in the project's `BUILD` file(s).

4. In the project's `BUILD` files, configure package visibility as desired.
