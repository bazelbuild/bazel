Project: /_project.yaml
Book: /_book.yaml

# Migrating from Xcode to Bazel

{% include "_buttons.html" %}

This page describes how to build or test an Xcode project with Bazel. It
describes the differences between Xcode and Bazel, and provides the steps for
converting an Xcode project to a Bazel project. It also provides troubleshooting
solutions to address common errors.

## Differences between Xcode and Bazel {:#dif-xcode-bazel}

*   Bazel requires you to explicitly specify every build target and its
    dependencies, plus the corresponding build settings via build rules.

*   Bazel requires all files on which the project depends to be present within
    the workspace directory or specified as dependencies in the `MODULE.bazel`
    file.

*   When building Xcode projects with Bazel, the `BUILD` file(s) become the
    source of truth. If you work on the project in Xcode, you must generate a
    new version of the Xcode project that matches the `BUILD` files using
    [rules_xcodeproj](https://github.com/buildbuddy-io/rules_xcodeproj/){: .external}
    whenever you update the `BUILD` files. Certain changes to the `BUILD` files
    such as adding dependencies to a target don't require regenerating the
    project which can speed up development. If you're not using Xcode, the
    `bazel build` and `bazel test` commands provide build and test capabilities
    with certain limitations described later in this guide.

## Before you begin {:#before-you-begin}

Before you begin, do the following:

1.  [Install Bazel](/install) if you have not already done so.

2.  If you're not familiar with Bazel and its concepts, complete the [iOS app
    tutorial](/start/ios-app)). You should understand the Bazel workspace,
    including the `MODULE.bazel` and `BUILD` files, as well as the concepts of
    targets, build rules, and Bazel packages.

3.  Analyze and understand the project's dependencies.

### Analyze project dependencies {:#analyze-project-dependencies}

Unlike Xcode, Bazel requires you to explicitly declare all dependencies for
every target in the `BUILD` file.

For more information on external dependencies, see [Working with external
dependencies](/docs/external).

## Build or test an Xcode project with Bazel {:#build-xcode-project}

To build or test an Xcode project with Bazel, do the following:

1.  [Create the `MODULE.bazel` file](#create-workspace)

2.  [(Experimental) Integrate SwiftPM dependencies](#integrate-swiftpm)

3.  [Create a `BUILD` file:](#create-build-file)

    a.  [Add the application target](#add-app-target)

    b.  [(Optional) Add the test target(s)](#add-test-target)

    c.  [Add the library target(s)](#add-library-target)

4.  [(Optional) Granularize the build](#granularize-build)

5.  [Run the build](#run-build)

6.  [Generate the Xcode project with rules_xcodeproj](#generate-the-xcode-project-with-rules_xcodeproj)

### Step 1: Create the `MODULE.bazel` file {:#create-workspace}

Create a `MODULE.bazel` file in a new directory. This directory becomes the
Bazel workspace root. If the project uses no external dependencies, this file
can be empty. If the project depends on files or packages that are not in one of
the project's directories, specify these external dependencies in the
`MODULE.bazel` file.

Note: Place the project source code within the directory tree containing the
`MODULE.bazel` file.

### Step 2: (Experimental) Integrate SwiftPM dependencies {:#integrate-swiftpm}

To integrate SwiftPM dependencies into the Bazel workspace with
[swift_bazel](https://github.com/cgrindel/swift_bazel){: .external}, you must
convert them into Bazel packages as described in the [following
tutorial](https://chuckgrindel.com/swift-packages-in-bazel-using-swift_bazel/){: .external}
.

Note: SwiftPM support is a manual process with many variables. SwiftPM
integration with Bazel has not been fully verified and is not officially
supported.

### Step 3: Create a `BUILD` file {:#create-build-file}

Once you have defined the workspace and external dependencies, you need to
create a `BUILD` file that tells Bazel how the project is structured. Create the
`BUILD` file at the root of the Bazel workspace and configure it to do an
initial build of the project as follows:

*   [Step 3a: Add the application target](#step-3a-add-the-application-target)
*   [Step 3b: (Optional) Add the test target(s)](#step-3b-optional-add-the-test-target-s)
*   [Step 3c: Add the library target(s)](#step-3c-add-the-library-target-s)

**Tip:** To learn more about packages and other Bazel concepts, see [Workspaces,
packages, and targets](/concepts/build-ref).

#### Step 3a: Add the application target {:#add-app-target}

Add a
[`macos_application`](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-macos.md#macos_application){: .external}
or an
[`ios_application`](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-ios.md#ios_application){: .external}
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

#### Step 3b: (Optional) Add the test target(s) {:#add-test-target}

Bazel's [Apple build
rules](https://github.com/bazelbuild/rules_apple){: .external} support running
unit and UI tests on all Apple platforms. Add test targets as follows:

*   [`macos_unit_test`](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-macos.md#macos_unit_test){: .external}
    to run library-based and application-based unit tests on a macOS.

*   [`ios_unit_test`](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-ios.md#ios_unit_test){: .external}
    to build and run library-based unit tests on iOS.

*   [`ios_ui_test`](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-ios.md#ios_ui_test){: .external}
    to build and run user interface tests in the iOS simulator.

*   Similar test rules exist for
    [tvOS](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-tvos.md){: .external},
    [watchOS](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-watchos.md){: .external}
    and
    [visionOS](https://github.com/bazelbuild/rules_apple/blob/master/doc/rules-visionos.md){: .external}.

At the minimum, specify a value for the `minimum_os_version` attribute. While
other packaging attributes, such as `bundle_identifier` and `infoplists`,
default to most commonly used values, ensure that those defaults are compatible
with the project and adjust them as necessary. For tests that require the iOS
simulator, also specify the `ios_application` target name as the value of the
`test_host` attribute.

#### Step 3c: Add the library target(s) {:#add-library-target}

Add an [`objc_library`](/reference/be/objective-c#objc_library) target for each
Objective-C library and a
[`swift_library`](https://github.com/bazelbuild/rules_swift/blob/master/doc/rules.md#swift_library){: .external}
target for each Swift library on which the application and/or tests depend.

Add the library targets as follows:

*   Add the application library targets as dependencies to the application
    targets.

*   Add the test library targets as dependencies to the test targets.

*   List the implementation sources in the `srcs` attribute.

*   List the headers in the `hdrs` attribute.

Note: You can use the [`glob`](/reference/be/functions#glob) function to include
all sources and/or headers of a certain type. Use it carefully as it might
include files you do not want Bazel to build.

You can browse existing examples for various types of applications directly in
the [rules_apple examples
directory](https://github.com/bazelbuild/rules_apple/tree/master/examples/). For
example:

*   [macOS application targets](https://github.com/bazelbuild/rules_apple/tree/master/examples/macos){: .external}

*   [iOS applications targets](https://github.com/bazelbuild/rules_apple/tree/master/examples/ios){: .external}

*   [Multi platform applications (macOS, iOS, watchOS, tvOS)](https://github.com/bazelbuild/rules_apple/tree/master/examples/multi_platform){: .external}

For more information on build rules, see [Apple Rules for
Bazel](https://github.com/bazelbuild/rules_apple){: .external}.

At this point, it is a good idea to test the build:

`bazel build //:<application_target>`

### Step 4: (Optional) Granularize the build {:#granularize-build}

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

*   The `glob()` function does not cross package boundaries, so as the number of
    packages grows the files matched by `glob()` will shrink.

*   When adding a `BUILD` file to a `main` directory, also add a `BUILD` file to
    the corresponding `test` directory.

*   Enforce healthy visibility limits across packages.

*   Build the project after each major change to the `BUILD` files and fix build
    errors as you encounter them.

### Step 5: Run the build {:#run-build}

Run the fully migrated build to ensure it completes with no errors or warnings.
Run every application and test target individually to more easily find sources
of any errors that occur.

For example:

```posix-terminal
bazel build //:my-target
```

### Step 6: Generate the Xcode project with rules_xcodeproj {:#generate-the-xcode-project-with-rules_xcodeproj}

When building with Bazel, the `MODULE.bazel` and `BUILD` files become the source
of truth about the build. To make Xcode aware of this, you must generate a
Bazel-compatible Xcode project using
[rules_xcodeproj](https://github.com/buildbuddy-io/rules_xcodeproj#features){: .external}
.

### Troubleshooting {:#troubleshooting}

Bazel errors can arise when it gets out of sync with the selected Xcode version,
like when you apply an update. Here are some things to try if you're
experiencing errors with Xcode, for example "Xcode version must be specified to
use an Apple CROSSTOOL".

*   Manually run Xcode and accept any terms and conditions.

*   Use Xcode select to indicate the correct version, accept the license, and
    clear Bazel's state.

```posix-terminal
  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

  sudo xcodebuild -license

  bazel sync --configure
```

*   If this does not work, you may also try running `bazel clean --expunge`.

Note: If you've saved your Xcode to a different path, you can use `xcode-select
-s` to point to that path.