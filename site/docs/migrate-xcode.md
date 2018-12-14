---
layout: documentation
title: Migrating from Xcode to Bazel
---

# Migrating from Xcode to Bazel

This guide describes how to build or test an Xcode project with Bazel. It
describes the differences between Xcode and Bazel, and provides the steps for
converting an Xcode project to a Bazel project.

## Contents

- [Differences between Xcode and Bazel](#differences-between-xcode-and-bazel)
- [Before you begin](#before-you-begin)
   - [Analyze project dependencies](#analyze-project-dependencies)
- [Build or test an Xcode project with Bazel](#build-or-test-an-xcode-project-with-bazel)
   - [Step 1: Create the `WORKSPACE` file](#step-1-create-the-workspace-file)
   - [Step 2: (Experimental) Integrate CocoaPod dependencies](#step-2-experimental-integrate-cocoapods-dependencies)
   - [Step 3: Create a `BUILD` file:](#step-3-create-a-build-file)
      - [Step 3a: Add the application target](#step-3a-add-the-application-target)
      - [Step 3b: (Optional) Add the test target(s)](#step-3b-optional-add-the-test-target-s)
      - [Step 3c: Add the library target(s)](#step-3c-add-the-library-target-s)
   - [Step 4: (Optional) Granularize the build](#step-4-optional-granularize-the-build)
   - [Step 5: Run the build](#step-5-run-the-build)
   - [Step 6: Generate the Xcode project with Tulsi](#step-6-generate-the-xcode-project-with-tulsi)

## Differences between Xcode and Bazel

*   Bazel requires you to explicitly specify every build target and its
    dependencies, plus the corresponding build settings via build rules.

*   Bazel requires all files on which the project depends to be present
    within the workspace directory or specified as imports in the `WORKSPACE`
    file.

*   When building Xcode projects with Bazel, the `BUILD` file(s) become the
    source of truth. If you work on the project in Xcode, you must generate a
    new version of the Xcode project that matches the `BUILD` files using
    [Tulsi](http://tulsi.bazel.build/) whenever you update the `BUILD` files. If
    you're not using Xcode, the `bazel build` and `bazel test` commands provide
    build and test capabilities with certain limitations described later in this
    guide.

*   Due to differences in build configuration schemas, such as directory layouts
    or build flags, Xcode might not be fully aware of the "big picture" of the
    build and thus some Xcode features might not work. Namely:

    *   Depending on the targets you select for conversion in Tulsi, Xcode might
        not be able to properly index the project source. This affects code
        completion and navigation in Xcode, since Xcode won't be able to see all
        of the project's source code.

    *   Static analysis, address sanitizers, and thread sanitizers might not
        work, since Bazel does not produce the outputs that Xcode expects for
        those features.

    *   If you generate an Xcode project with Tulsi and use that project to run
        tests from within Xcode, Xcode will run the tests instead of
        Bazel. To run tests with Bazel, run the `bazel test` command manually.

## Before you begin

Before you begin, do the following:

1.  [Install Bazel](install.html) if
    you have not already done so.

2.  If you're not familiar with Bazel and its concepts, complete the
    [iOS app tutorial](tutorial/ios-app.html).
    You should understand the Bazel workspace, including the `WORKSPACE` and
    `BUILD` files, as well as the concepts of targets, build rules, and Bazel
    packages.

3.  Analyze and understand the project's dependencies.

### Analyze project dependencies

Unlike Xcode, Bazel requires you to explicitly declare all dependencies for
every target in the `BUILD` file.

For more information on external dependencies, see
[Working with external dependencies](external.html).

## Build or test an Xcode project with Bazel

To build or test an Xcode project with Bazel, do the following:

1.  [Create the `WORKSPACE` file](#step-1-create-the-workspace-file)

2. [(Experimental) Integrate CocoaPods dependencies](#step-2-experimental-integrate-cocoapods-dependencies)

3.  [Create a `BUILD` file:](#step-3-create-a-build-file)

    a.  [Add the application target](#step-3a-add-the-application-target)

    b.  [(Optional) Add the test target(s)](#step-3b-optional-add-the-test-target-s)

    c.  [Add the library target(s)](#step-3c-add-the-library-target-s)

4.  [(Optional) Granularize the build](#step-4-optional-granularize-the-build)

5.  [Run the build](#step-5-run-the-build)

6.  [Generate the Xcode project with Tulsi](#step-6-generate-the-xcode-project-with-tulsi)

### Step 1: Create the `WORKSPACE` file

Create a `WORKSPACE` file in a new directory. This directory becomes the Bazel
workspace root. If the project uses no external dependencies, this file can be
empty. If the project depends on files or packages that are not in one of the
project's directories, specify these external dependencies in the `WORKSPACE`
file.

**Note:** Place the project source code within the directory tree containing the
          `WORKSPACE` file.

### Step 2: (Experimental) Integrate CocoaPods dependencies

To integrate CocoaPods dependencies into the Bazel workspace, you must convert
them into Bazel packages as described in [Converting CocoaPods dependencies](migrate-cocoapods.md).

**Note:** CocoaPods conversion is a manual process with many variables.
CocoaPods integration with Bazel has not been fully verified and is not
officially supported.


### Step 3: Create a `BUILD` file

Once you have defined the workspace and external dependencies, you need to
create a `BUILD` file that tells Bazel how the project is structured. Create
the `BUILD` file at the root of the Bazel workspace and configure it to do an
initial build of the project as follows:

*  [Step 3a: Add the application target](#step-3a-add-the-application-target)
*  [Step 3b: (Optional) Add the test target(s)](#step-3b-optional-add-the-test-target-s)
*  [Step 3c: Add the library target(s)](#step-3c-add-the-library-target-s)

**Tip:** To learn more about packages and other Bazel concepts, see
[Bazel Terminology](build-ref.html).

#### Step 3a: Add the application target

Add a [`macos_application`](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-macos.md#macos_application)
or an [`ios_application`](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-ios.md#ios_application)
rule target. This target builds a macOS or iOS application bundle, respectively.
In the target, specify the following at the minimum:

*   `bundle_id` - the bundle ID (reverse-DNS path followed by app name) of the
    binary.

*   `provisioning_profile` - provisioning profile from your Apple Developer
    account (if building for an iOS device device).

*   `families` (iOS only) - whether to build the application for iPhone, iPad,
    or both.

*   `infoplists` - list of .plist files to merge into the final Info.plist file.

*   `minimum_os_version` - the minimum version of macOS or iOS that the
    application supports. This ensures Bazel builds the application with the
    correct API levels.

#### Step 3b: (Optional) Add the test target(s)

Bazel's [Apple build rules](https://github.com/bazelbuild/rules_apple) support
running library-based unit tests on iOS and macOS, as well as application-based
tests on macOS. For application-based tests on iOS or UI tests on either
platform, Bazel will build the test outputs but the tests must run within Xcode
through a project generated with Tulsi. Add test targets as follows:

*   [`macos_unit_test`](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-macos.md#macos_unit_test)
    to run library-based and application-based unit tests on a macOS.

*   [`ios_unit_test`](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-ios.md#ios_unit_test)
    to run library-based unit tests on iOS. For tests requiring the iOS
    simulator, Bazel will build the test outputs but not run the tests. You must
    [generate an Xcode project with Tulsi](#step-5-generate-the-xcode-project-with-tulsi)
    and run the tests from within Xcode.

*   [`ios_ui_test`](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-ios.md#ios_ui_test)
    to build outputs required to run user interface tests in the iOS simulator
    using Xcode. You must [generate an Xcode project with Tulsi](#step-5-generate-the-xcode-project-with-tulsi)
    and run the tests from within Xcode. Bazel cannot natively run UI tests.

At the minimum, specify a value for the `minimum_os_version` attribute. While
other packaging attributes, such as `bundle_identifier` and `infoplists`,
default to most commonly used values, ensure that those defaults are compatible
with the project and adjust them as necessary. For tests that require the iOS
simulator, also specify the `ios_application` target name as the value of the
`test_host` attribute.


#### Step 3c: Add the library target(s)

Add an [`objc_library`](be/objective-c.html#objc_library)
target for each Objective C library and a [`swift_library`](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-swift.md)
target for each Swift library on which the application and/or tests depend.


Add the library targets as follows:

*   Add the application library targets as dependencies to the application
    targets.

*   Add the test library targets as dependencies to the test targets.

*   List the implementation sources in the `srcs` attribute.

*   List the headers in the `hdrs` attribute.

**Note:** You can use the [`glob`](be/functions.html#glob)
function to include all sources and/or headers of a certain type. Use it
carefully as it might include files you do not want Bazel to build.

For more information on build rules, see [Apple Rules for Bazel](https://github.com/bazelbuild/rules_apple).

At this point, it is a good idea to test the build:

`bazel build //:<application_target>`

### Step 4: (Optional) Granularize the build

If the project is large, or as it grows, consider chunking it into multiple
Bazel packages. This increased granularity provides:

*   Increased incrementality of builds,

*   Increased parallelization of build tasks,

*   Better maintainability for future users,

*   Better control over source code visibility across targets and packages. This
    prevents issues such as libraries containing implementation details leaking
    into public APIs.

Tips for granularizing the project:

*   Put each library in its own Bazel package. Start with those requiring the
    fewest dependencies and work your way up the dependency tree.

*   As you add `BUILD` files and specify targets, add these new targets to the
   `deps` attributes of targets that depend on them.

*   The `glob()` function does not cross package boundaries, so as the number
    of packages grows the files matched by `glob()` will shrink.

*   When adding a `BUILD` file to a `main` directory, also add a `BUILD` file to
    the corresponding `test` directory.

*   Enforce healthy visibility limits across packages.

*   Build the project after each major change to the `BUILD` files and fix
    build errors as you encounter them.

### Step 5: Run the build

Run the fully migrated build to ensure it completes with no errors or warnings.
Run every application and test target individually to more easily find sources
of any errors that occur.

For example:

```bash
bazel build //:my-target
```

### Step 6: Generate the Xcode project with Tulsi

When building with Bazel, the `WORKSPACE` and `BUILD` files become the source
of truth about the build. To make Xcode aware of this, you must generate a
Bazel-compatible Xcode project using [Tulsi](http://tulsi.bazel.build/).
