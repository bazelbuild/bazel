---
layout: documentation
title: Tutorial - Build an iOS App
---

# Tutorial - Build an iOS App

Like the [Android app](android-app.md) you built in the previous step, the iOS
app is a simple mobile app that communicates with the
[backend server](backend-server.md).

Here, you'll do the following:

*   Review the source files for the app
*   Create a `BUILD` file
*   Build the app for the simulator
*   Find the build outputs
*   Run/Debug the app on the simulator
*   Build the app for a device
*   Install the app on a device

Note that, unlike with the Android app, you don't have to modify your
`WORKSPACE` file to add iOS-specific external dependencies.

If you're following the steps in this tutorial on Mac OS X, you can go ahead
and build the sample iOS app as described below. If you are on Linux, skip ahead
to the [next step](backend-server.md).

## Review the source files

Let's take a look at the source files for the app. These are located in
`$WORKSPACE/ios-app/UrlGet`. Again, you're just looking at these files now to
become familiar with the structure of the app. You don't have to edit any of the
source files to complete this tutorial.

## Create a BUILD file

At a command-line prompt, open your new `BUILD` file for editing:

```bash
$ vi $WORKSPACE/ios-app/BUILD
```

## Add an objc_library rule

Bazel provides several build rules that you can use to build an app for the
iOS platform. For this tutorial, you'll first use the
[`objc_library`](/docs/be/objective-c.html#objc_library) rule to tell Bazel
how to build an
[static library](https://developer.apple.com/library/ios/technotes/iOSStaticLibraries/Introduction.html)
from the app source code and Xib files. Then you'll use the
`objc_binary` rule to tell it how to bundle the iOS application. (Note that
this is a minimal use case of the Objective-C rules in Bazel. For example, you
have to use the `ios_application` rule to build multi-architecture iOS
apps.)

Add the following to your `BUILD` file:

```python
objc_library(
    name = "UrlGetClasses",
    srcs = [
        "UrlGet/AppDelegate.m",
        "UrlGet/UrlGetViewController.m",
    ],
    hdrs = glob(["UrlGet/*.h"]),
    xibs = ["UrlGet/UrlGetViewController.xib"],
)
```

Note the name of the rule, `UrlGetClasses`.

## Add an objc_binary rule

The [`objc_binary`](/docs/be/objective-c.html#objc_binary) rule creates a
binary to be bundled in the application.

Add the following to your `BUILD` file:

```python
objc_binary(
    name = "ios-app-binary",
    srcs = [
        "UrlGet/main.m",
    ],
    deps = [
        ":UrlGetClasses",
    ],
)

```
Note how the `deps` attribute references the output of the
`UrlGetClasses` rule you added to the `BUILD` file above.

## Add an ios_application rule

The [`ios_application`](/docs/be/objective-c.html#ios_application) rule
creates the bundled `.ipa` archive file for the application and also generates
an Xcode project file.

Add the following to your `BUILD` file:

```python
ios_application(
    name = "ios-app",
    binary = ":ios-app-binary",
    infoplist = "UrlGet/UrlGet-Info.plist",
)
```

Now, save and close the file. You can compare your `BUILD` file to the
[completed example](https://github.com/bazelbuild/examples/blob/master/tutorial/ios-app/BUILD)
in the `master` branch of the GitHub repo.

## Build the app for the simulator

Make sure that your current working directory is inside your Bazel workspace:

```bash
$ cd $WORKSPACE
```

Now, enter the following to build the sample app:

```bash
$ bazel build //ios-app:ios-app
```

Bazel now launches and builds the sample app. During the build process, its
output will appear similar to the following:

```bash
INFO: Found 1 target...
Target //ios-app:ios-app up-to-date:
  bazel-bin/ios-app/ios-app.ipa
  bazel-bin/ios-app/ios-app.xcodeproj/project.pbxproj
INFO: Elapsed time: 3.765s, Critical Path: 3.44s
```

## Find the build outputs

The `.ipa` file and other outputs are located in the
`$WORKSPACE/bazel-bin/ios-app` directory.

## Run/Debug the app on the simulator

You can now run the app from Xcode using the iOS Simulator. To run the app,
open the project directory `$WORKSPACE/bazel-bin/ios-app/ios-app.xcodeproj` in
Xcode, choose an iOS Simulator as the runtime scheme and then click the **Run**
button.

**Note:** If you change anything about the project file set in Xcode (for
example, if you add or remove a file, or add or change a dependency), you must
rebuild the app using Bazel and then re-open the project.

## Build the app for a device

You need to set up bazel so that it can find the appropriate provisioning
profile for the device you want to build for. To set up the "default"
provisioning profile for all bazel builds:

   1. Go to [Apple Profiles](https://developer.apple.com/account/ios/profile/profileList.action)
      and download the appropriate provisioning profile for your device.
      If this is confusing, please refer to [Apple's documentation](https://developer.apple.com/library/ios/documentation/IDEs/Conceptual/AppDistributionGuide/MaintainingProfiles/MaintainingProfiles.html).
   1. Move your profile into `$WORKSPACE/tools/objc`.
   1. Optional - You may want to add your profile to your `.gitignore`.
   1. Edit `$WORKSPACE/tools/objc/BUILD` and add:

      ```python
      filegroup(
          name = "default_provisioning_profile",
          srcs = ["<NAME OF YOUR PROFILE>.mobileprovision"],
      )
      ```

Now you should be able to build the app for your device:

```bash
$ bazel build //ios-app:ios-app --ios_multi_cpus=armv7,arm64
```

This will build the app "fat". If you would prefer just to build for
your specific device architecture you can designate a single architecture.

If you would like to select a specific Xcode version/SDK version you can do so
with the `--xcode_version=7.2 --ios_sdk_version=9.2` options. Make sure that
the Xcode version that you select has the appropriate SDK installed in it.

If you would like to specify a minimum version of iOS to run against, you can
do so with the `--ios_minimum_os=7.0` option.

## Install the app on a device

The easiest way to install the app on the device is to launch Xcode and use the
`Windows > Devices` command. Select your plugged in device from the list on the
left, and then add the app by clicking on the "plus" sign under installed apps
and selecting the `.ipa` that you built.

If your app does not launch, please make sure that your device was on your
provisioning profile. The `View Device Logs` button on the `Devices` screen in
Xcode may provide other information as to what has gone wrong.

## What's next

The next step is to build a [backend server](backend-server.md) for the two
mobile apps you built in this tutorial.
