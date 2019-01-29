---
layout: documentation
title: Build Tutorial - iOS
---

# Introduction to Bazel: Building an iOS App

In this tutorial, you will learn how to build a simple iOS app. You'll do the
following:

* ToC
{:toc}


## Set up your environment

To get started, install Bazel and Xcode, and get the sample project.

### Install Bazel

Follow the [installation instructions](../install.md) to install Bazel and
its dependencies.

### Install Xcode

Download and install [Xcode](https://developer.apple.com/xcode/downloads/).
Xcode contains the compilers, SDKs, and other tools required by Bazel to build
Apple applications.

### Get the sample project

You also need to get the sample project for the tutorial from GitHub. The GitHub
repo has two branches: `source-only` and `master`. The `source-only` branch
contains the source files for the project only. You'll use the files in this
branch in this tutorial. The `master` branch contains both the source files
and completed Bazel `WORKSPACE` and `BUILD` files. You can use the files in this
branch to check your work when you've completed the tutorial steps.

Enter the following at the command line to get the files in the `source-only`
branch:

```bash
cd $HOME
git clone -b source-only https://github.com/bazelbuild/examples
```

The `git clone` command creates a directory named `$HOME/examples/`. This
directory contains several sample projects for Bazel. The project files for this
tutorial are in `$HOME/examples/tutorial/ios-app`.

## Set up a workspace

A [workspace](../build-ref.html#workspaces) is a directory that contains the
source files for one or more software projects, as well as a `WORKSPACE` file
and `BUILD` files that contain the instructions that Bazel uses to build
the software. The workspace may also contain symbolic links to output
directories.

A workspace directory can be located anywhere on your filesystem and is denoted
by the presence of the `WORKSPACE` file at its root. In this tutorial, your
workspace directory is `$HOME/examples/tutorial/`, which contains the sample
project files you cloned from the GitHub repo in the previous step.

Note that Bazel itself doesn't impose any requirements for organizing source
files in your workspace. The sample source files in this tutorial are organized
according to conventions for the target platform.

For your convenience, set the `$WORKSPACE` environment variable now to refer to
your workspace directory. At the command line, enter:

```bash
export WORKSPACE=$HOME/examples/tutorial
```

### Create a WORKSPACE file

Every workspace must have a text file named `WORKSPACE` located in the top-level
workspace directory. This file may be empty or it may contain references
to [external dependencies](../external.html) required to build the
software.

For now, you'll create an empty `WORKSPACE` file, which simply serves to
identify the workspace directory. In later steps, you'll update the file to add
external dependency information.

Enter the following at the command line:

```bash
touch $WORKSPACE/WORKSPACE
open -a Xcode $WORKSPACE/WORKSPACE
```

This creates and opens the empty `WORKSPACE` file.

### Update the WORKSPACE file

To build applications for Apple devices, Bazel needs to pull the latest
[Apple build rules](https://github.com/bazelbuild/rules_apple) from its GitHub
repository. To enable this, add the following [`git_repository`](../be/workspace.html#git_repository)
rules to your `WORKSPACE` file:

```
git_repository(
    name = "build_bazel_rules_apple",
    remote = "https://github.com/bazelbuild/rules_apple.git",
    tag = "0.4.0",
)
git_repository(
    name = "bazel_skylib",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    tag = "0.3.1",
)
```

**NOTE:** "Always use the [latest version of the build_apple rules](https://github.com/bazelbuild/rules_apple/releases)
in the `tag` attribute. Make sure to check the latest dependencies required in
`rules_apple`'s [project](https://github.com/bazelbuild/rules_apple)."

**NOTE:** You **must** set the value of the `name` attribute in the
`git_repository` rule to `build_bazel_rules_apple` or the build will fail.

## Review the source files

Take a look at the source files for the app located in
`$WORKSPACE/ios-app/UrlGet`. Again, you're just looking at these files now to
become familiar with the structure of the app. You don't have to edit any of the
source files to complete this tutorial.

## Create a BUILD file

At a command-line prompt, open a new `BUILD` file for editing:

```bash
touch $WORKSPACE/ios-app/BUILD
open -a Xcode $WORKSPACE/ios-app/BUILD
```

### Add the rule load statement

To build iOS targets, Bazel needs to load build rules from its GitHub repository
whenever the build runs. To make these rules available to your project, add the
following load statement to the beginning of your `BUILD` file:

```
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
```

We only need to load the `ios_application` rule because the `objc_library` rule
is built into the Bazel package.

### Add an objc_library rule

Bazel provides several build rules that you can use to build an app for the
iOS platform. For this tutorial, you'll first use the
[`objc_library`](../be/objective-c.html#objc_library) rule to tell Bazel
how to build a static library from the app source code and Xib files. Then
you'll use the [`ios_application`](https://github.com/bazelbuild/rules_apple)
rule to tell it how to build the application binary and the `.ipa` bundle.

**NOTE:** This tutorial presents a minimal use case of the Objective-C rules in
Bazel. For example, you have to use the `ios_application` rule to build
multi-architecture iOS apps.

Add the following to your `BUILD` file:

```python
objc_library(
    name = "UrlGetClasses",
    srcs = [
         "UrlGet/AppDelegate.m",
         "UrlGet/UrlGetViewController.m",
         "UrlGet/main.m",
    ],
    hdrs = glob(["UrlGet/*.h"]),
    xibs = ["UrlGet/UrlGetViewController.xib"],
)
```

Note the name of the rule, `UrlGetClasses`.

### Add an ios_application rule

The [`ios_application`](../be/objective-c.html#ios_application) rule builds
the application binary and creates the `.ipa` bundle file.

Add the following to your `BUILD` file:

```python
ios_application(
    name = "ios-app",
    bundle_id = "Google.UrlGet",
    families = [
        "iphone",
        "ipad",
    ],
    minimum_os_version = "9.0",
    infoplists = [":UrlGet/UrlGet-Info.plist"],
    visibility = ["//visibility:public"],
    deps = [":UrlGetClasses"],
)
```

**NOTE:** Please update the `minimum_os_version` attribute to the minimum
version of iOS that you plan to support.

Note how the `deps` attribute references the output of the `UrlGetClasses` rule
you added to the `BUILD` file above.

Now, save and close the file. You can compare your `BUILD` file to the
[completed example](https://github.com/bazelbuild/examples/blob/master/tutorial/ios-app/BUILD)
in the `master` branch of the GitHub repo.

## Build and deploy the app

You are now ready to build your app and deploy it to a simulator and onto an
iOS device.

**NOTE:** The app launches standalone but requires a backend server in order to
produce output. See the README file in the sample project directory to find out
how to build the backend server.

### Build the app for the simulator

Make sure that your current working directory is inside your Bazel workspace:

```bash
cd $WORKSPACE
```

Now, enter the following to build the sample app:

```bash
bazel build //ios-app:ios-app
```

Bazel launches and builds the sample app. During the build process, its
output will appear similar to the following:

```bash
INFO: Found 1 target...
Target //ios-app:ios-app up-to-date:
  bazel-bin/ios-app/ios-app.ipa
INFO: Elapsed time: 0.565s, Critical Path: 0.44s
```

### Find the build outputs

The `.ipa` file and other outputs are located in the
`$WORKSPACE/bazel-bin/ios-app` directory.

### Run and debug the app in the simulator

You can now run the app from Xcode using the iOS Simulator. First, [generate an Xcode project using Tulsi](http://tulsi.bazel.io/).

Then, open the project in Xcode, choose an iOS Simulator as the runtime scheme,
and click **Run**.

**Note:** If you modify any project files in Xcode (for example, if you add or
remove a file, or add or change a dependency), you must rebuild the app using
Bazel, re-generate the Xcode project in Tulsi, and then re-open the project in
Xcode.

### Build the app for a device

To build your app so that it installs and launches on an iOS device, Bazel needs
the appropriate provisioning profile for that device model. Do the following:

1. Go to your [Apple Developer Account](https://developer.apple.com/account) and
   download the appropriate provisioning profile for your device. See
   [Apple's documentation](https://developer.apple.com/library/ios/documentation/IDEs/Conceptual/AppDistributionGuide/MaintainingProfiles/MaintainingProfiles.html)
   for more information.

2. Move your profile into `$WORKSPACE`.

3. (Optional) Add your profile to your `.gitignore` file.

4. Add the following line to the `ios_application` target in your `BUILD` file:

   ```python
   provisioning_profile = "<your_profile_name>.mobileprovision",
   ```

   **NOTE:** Ensure the profile is correct so that the app can be installed on
   a device.

Now build the app for your device:

```bash
bazel build //ios-app:ios-app --ios_multi_cpus=armv7,arm64
```

This builds the app as a fat binary. To build for a specific device
architecture, designate it in the build options.

To build for a specific Xcode version, use the `--xcode_version` option. To
build for a specific SDK version, use the `--ios_sdk_version` option. The
`--xcode_version` option is sufficient in most scenarios.

To specify a minimum required iOS version, add the `minimum_os_version`
parameter to the `ios_application` build rule in your `BUILD` file.

You can also use [Tulsi](http://tulsi.bazel.io/docs/gettingstarted.html) to
build your app using a GUI rather than the command line.

### Install the app on a device

The easiest way to install the app on the device is to launch Xcode and use the
`Windows > Devices` command. Select your plugged-in device from the list on the
left, then add the app by clicking the **Add** (plus sign) button under
"Installed Apps" and selecting the `.ipa` file that you built.

If your app fails to install on your device, ensure that you are specifying the
correct provisioning profile in your `BUILD` file (step 4 in the previous
section).

If your app fails to launch, make sure that your device is part of your
provisioning profile. The `View Device Logs` button on the `Devices` screen in
Xcode may provide other information as to what has gone wrong.

## Review your work

In this tutorial, you used Bazel to build an iOS app. To accomplish that, you:

*   Set up your environment by installing Bazel and Xcode, and downloading the
    sample project
*   Set up a Bazel [workspace](workspace.md) that contained the source code
    for the app and a `WORKSPACE` file that identifies the top level of the
    workspace directory
*   Updated the `WORKSPACE` file to contain references to the required
    external dependencies
*   Created a `BUILD` file
*   Ran Bazel to build the app for the simulator and an iOS device
*   Ran the app in the simulator and on an iOS device

The built app is located in the `$WORKSPACE/bazel-bin` directory.

Completed `WORKSPACE` and `BUILD` files for this tutorial are located in the
[master branch](https://github.com/bazelbuild/examples/tree/master/tutorial)
of the GitHub repo. You can compare your work to the completed files for
additional help or troubleshooting.
