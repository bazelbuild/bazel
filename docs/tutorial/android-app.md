---
layout: documentation
title: Tutorial - Build an Android App
---

# Tutorial - Build an Android App

The sample Android app in this tutorial is a very simple application that makes
an HTTP connection to the [backend server](backend-server.md) and displays the
resulting response.

Here, you'll do the following:

*   Review the source files for the app
*   Update the `WORKSPACE` file
*   Create a `BUILD` file
*   Run the build
*   Find the build outputs
*   Run the app

## Review the source files

Let's take a look at the source files for the app. These are located in
`$WORKSPACE/android/`.

The key files and directories are:

<table class="table table-condensed table-striped">
<thead>
<tr>
<td>Name</td>
<td>Location</td>
</tr>
</thead>
<tbody>
<tr>
<td>Manifest file</td>
<td><code>src/main/java/com/google/bazel/example/android/AndroidManifest.xml</code></td>
</tr>
<tr>
<td>Activity source file</td>
<td><code>src/main/java/com/google/bazel/example/android/activities/MainActivity.java</code></td>
</tr>
<tr>
<td>Resource file directory</td>
<td><code>src/main/java/com/google/bazel/example/android/res/</code></td>
</tr>
</tbody>
</table>

Note that you're just looking at these files now to become familiar with the
structure of the app. You don't have to edit any of the source files to complete
this tutorial.

## Update the WORKSPACE file

Bazel needs to run the Android SDK
[build tools](https://developer.android.com/tools/revisions/build-tools.html)
and uses the SDK libraries to build the app. This means that you need to add
some information to your `WORKSPACE` file so that Bazel knows where to find
them.  Note that this step is not required when you build for other platforms.
For example, Bazel automatically detects the location of Java, C++ and
Objective-C compilers from settings in your environment.

Add the following lines to your `WORKSPACE` file:

```python
android_sdk_repository(
    name = "androidsdk",
    # Replace with path to Android SDK on your system
    path = "/Users/username/Library/Android/sdk",
    # Replace with the Android SDK API level
    api_level = 23,
    # Replace with the verion in sdk/build-tools/
    build_tools_version="23.0.0"
)
```

**Optional:** This is not required by this tutorial, but if you want to compile
native code into your Android app, you also need to download the
[Android NDK](https://developer.android.com/ndk/downloads/index.html) and
tell Bazel where to find it by adding the following rule to your `WORKSPACE`
file:

```python
android_ndk_repository(
    name = "androidndk",
    # Replace with path to Android NDK on your system
    path = "/Users/username/Library/Android/ndk",
    # Replace with the Android NDK API level
    api_level = 21
)
```

`api_level` is the version of the Android API the SDK and the NDK target
(for example, 19 for Android K and 21 for Android L). It's not necessary to set
the API levels to the same value for the SDK and NDK.
[This web page](https://developer.android.com/ndk/guides/stable_apis.html)
contains a map from Android releases to NDK-supported API levels.

## Create a BUILD file

A [`BUILD` file](/docs/build-ref.html#BUILD_files) is a text file that describes
the relationship between a set of build outputs -- for example, compiled
software libraries or executables -- and their dependencies. These dependencies
may be source files in your workspace or other build outputs. `BUILD` files are
written in the Bazel *build language*.

`BUILD` files are part of concept in Bazel known as the *package hierarchy*.
The package hierarchy is a logical structure that overlays the directory
structure in your workspace. Each [package](/docs/build-ref.html#packages) is a
directory (and its subdirectories) that contains a related set of source files
and a `BUILD` file. The package also includes any subdirectories, excluding
those that contain their own `BUILD` file. The *package name* is the name of the
directory where the `BUILD` file is located.

Note that this package hierarchy is distinct from, but coexists with, the Java
package hierarchy for your Android app.

For the simple Android app in this tutorial, we'll consider all the source files
in `$WORKSPACE/android/` to comprise a single Bazel package. A more complex
project may have many nested packages.

At a command-line prompt, open your new `BUILD` file for editing:

```bash
$ vi $WORKSPACE/android/BUILD
```

### Add an android_library rule

A `BUILD` file contains several different types of instructions for Bazel. The
most important type is the [build rule](/docs/build-ref.html#funcs), which tells
Bazel how to build an intermediate or final software output from a set of source
files or other dependencies.

Bazel provides two build rules, `android_library` and `android_binary`, that you
can use to build an Android app. For this tutorial, you'll first use the
[`android_library`](/docs/be/android.html#android_library) rule to tell
Bazel how to build an
[Android library module](http://developer.android.com/tools/projects/index.html#LibraryProjects)
from the app source code and resource files. Then you'll use the
`android_binary` rule to tell it how to build the Android application package.

Add the following to your `BUILD` file:

```python
android_library(
  name = "activities",
  srcs = glob(["src/main/java/com/google/bazel/example/android/activities/*.java"]),
  custom_package = "com.google.bazel.example.android.activities",
  manifest = "src/main/java/com/google/bazel/example/android/activities/AndroidManifest.xml",
  resource_files = glob(["src/main/java/com/google/bazel/example/android/activities/res/**"]),
)
```

As you can see, the `android_library` build rule contains a set of attributes
that specify the information that Bazel needs to build a library module from the
source files. Note also that the name of the rule is `activities`. You'll
reference the rule using this name as a dependency in the `android_binary` rule.

### Add an android_binary rule

The [`android_binary`](/docs/be/android.html#android_binary) rule builds
the Android application package (`.apk` file) for your app.

Add the following to your build file:

```python
android_binary(
    name = "android",
    custom_package = "com.google.bazel.example.android",
    manifest = "src/main/java/com/google/bazel/example/android/AndroidManifest.xml",
    resource_files = glob(["src/main/java/com/google/bazel/example/android/res/**"]),
    deps = [":activities"],
)
```

Here, the `deps` attribute references the output of the `activities` rule you
added to the `BUILD` file above. This means that, when Bazel builds the output
of this rule, it checks first to see if the output of the `activities` library
rule has been built and is up-to-date. If not, it builds it and then uses that
output to build the application package file.

Now, save and close the file. You can compare your `BUILD` file to the
[completed example](https://github.com/bazelbuild/examples/blob/master/tutorial/android/BUILD)
in the `master` branch of the GitHub repo.

## Run the build

You use the
[`bazel`](/docs/bazel-user-manual.html) command-line tool to run builds, execute
unit tests and perform other operations in Bazel. This tool is located in the
`output` subdirectory of the location where you installed Bazel. During
[installation](/docs/install.md), you probably added this location to your
path.

Before you build the sample app, make sure that your current working directory
is inside your Bazel workspace:

```bash
$ cd $WORKSPACE
```

Now, enter the following to build the sample app:

```bash
$ bazel build //android:android
```

The [`build`](/docs/bazel-user-manual.html#build) subcommand instructs Bazel to
build the target that follows. The target is specified as the name of a build
rule inside a `BUILD` file, with along with the package path relative to
your workspace directory. Note that you can sometimes omit the package path
or target name, depending on your current working directory at the command
line and the name of the target. See [Labels](/docs/build-ref.html#labels) in
*Bazel Concepts and Terminology* page for more information about target labels
and paths.

Bazel now launches and builds the sample app. During the build process, its
output will appear similar to the following:

```bash
INFO: Found 1 target...
Target //android:android up-to-date:
  bazel-bin/android/android_deploy.jar
  bazel-bin/android/android_unsigned.apk
  bazel-bin/android/android.apk
INFO: Elapsed time: 7.237s, Critical Path: 5.81s
```

## Find the build outputs

Bazel stores the outputs of both intermediate and final build operations in
a set of per-user, per-workspace output directories. These directories are
symlinked from the following locations:

* `$WORKSPACE/bazel-bin`, which stores binary executables and other runnable
   build outputs
* `$WORKSPACE/bazel-genfiles`, which stores intermediary source files that are
   generated by Bazel rules
* `$WORKSPACE/bazel-out`, which stores other types of build outputs

Bazel stores the Android `.apk` file generated using the `android_binary` rule
in the `bazel-bin/android/` directory, where the subdirectory name `android` is
derived from the name of the Bazel package.

At a command prompt, list the contents of this directory and find the
`android.apk` file:

```bash
$ ls $WORKSPACE/bazel-bin/android
```

## Run the app

You can now deploy the app to a connected Android device or emulator from the
command line using the
[`bazel mobile-install`](http://bazel.io/docs/bazel-user-manual.html#mobile-install)
command. This command uses the Android Debug Bridge (`adb`) to communicate with
the device. You must set up your device to use `adb` following the instructions
in
[Android Debug Bridge](http://developer.android.com/tools/help/adb.html) before
deployment.

Enter the following:

```bash
$ bazel mobile-install //android:android
```

Note that the `mobile-install` subcommand also supports the
[`--incremental`](http://bazel.io/docs/bazel-user-manual.html#mobile-install)
flag that can be used to deploy only those parts of the app that have changed
since the last deployment.

## What's next

Now that you've built a sample app for Android, it's time to do the same for
the [iOS app](ios-app.md).
