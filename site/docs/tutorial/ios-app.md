---
layout: documentation
title: Tutorial - Build an iOS App
---

# Tutorial - Build an iOS App

Like the [Android app](android-app.md) you built in the previous step, the iOS
app is a simple mobile app that communicates with the [backend server](backend-server.md).

If you're following the steps in this tutorial on macOS, you can go ahead
and build the sample iOS app as described below. If you are on Linux, skip ahead
to the [next step](backend-server.md).

Here, you'll do the following:

*   Review the source files for the app
*   Create a `BUILD` file
*   Build the app for the simulator
*   Find the build outputs
*   Run/Debug the app on the simulator
*   Build the app for a device
*   Install the app on a device

## Update the WORKSPACE file

To build applications for Apple devices, Bazel needs to pull the latest
[Apple build rules](https://github.com/bazelbuild/rules_apple) from its GitHub
repository. To enable this, add the following to your `WORKSPACE` file:

```
git_repository(
    name = "build_bazel_rules_apple",
    remote = "https://github.com/bazelbuild/rules_apple.git",
    tag = "0.0.1",
)
```

## Review the source files

Let's take a look at the source files for the app. These are located in
`$WORKSPACE/ios-app/UrlGet`. Again, you're just looking at these files now to
become familiar with the structure of the app. You don't have to edit any of the
source files to complete this tutorial.

## Create a BUILD file

At a command-line prompt, open your new `BUILD` file for editing:

```bash
vi $WORKSPACE/ios-app/BUILD
```

## Add the rule load statement

To build iOS targets, Bazel needs to load build rules from its GitHub repository
whenever the build runs. To make these rules available to your project, add the
following load statement to the beginning of your `BUILD` file:

```
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
```

## Add an objc_library rule

Bazel provides several build rules that you can use to build an app for the
iOS platform. For this tutorial, you'll first use the
[`objc_library`](/docs/be/objective-c.html#objc_library) rule to tell Bazel
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

## Add an ios_application rule

The [`ios_application`](/docs/be/objective-c.html#ios_application) rule builds
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
    infoplists = [":UrlGet/UrlGet-Info.plist"],
    visibility = ["//visibility:public"],
    deps = [":UrlGetClasses"],
)
```

Note how the `deps` attribute references the output of the `UrlGetClasses` rule
you added to the `BUILD` file above.

Now, save and close the file. You can compare your `BUILD` file to the
[completed example](https://github.com/bazelbuild/examples/blob/master/tutorial/ios-app/BUILD)
in the `master` branch of the GitHub repo.

## Build the app for the simulator

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



## Find the build outputs

The `.ipa` file and other outputs are located in the
`$WORKSPACE/bazel-bin/ios-app` directory.

## Run/Debug the app on the simulator

You can now run the app from Xcode using the iOS Simulator. First, [generate an Xcode project using Tulsi](http://tulsi.bazel.io/).
Then, open the project in Xcode, choose an iOS Simulator as the runtime scheme,
and click **Run**.

**Note:** If you modify any project files in Xcode (for example, if you add or
remove a file, or add or change a dependency), you must rebuild the app using
Bazel, re-generate the Xcode project in Tulsi, and then re-open the project in
Xcode.

## Build the app for a device

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

## Install the app on a device

The easiest way to install the app on the device is to launch Xcode and use the
`Windows > Devices` command. Select your plugged-in device from the list on the
left, then add the app by clicking on the "plus" sign under installed apps and
selecting the `.ipa` that you built.

If your app fails to install on your device, ensure that you are specifying the
correct provisioning profile in your `BUILD` file (step 4 in the previous
section).

If your app fails to launch, make sure that your device is part of your
provisioning profile. The `View Device Logs` button on the `Devices` screen in
Xcode may provide other information as to what has gone wrong.

## What's next

The next step is to build a [backend server](backend-server.md) for the two
mobile apps you built in this tutorial.
