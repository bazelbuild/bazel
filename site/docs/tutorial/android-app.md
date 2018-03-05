---
layout: documentation
title: Build Tutorial - Android
---

Introduction to Bazel: Building an Android App
==========

In this tutorial, you will learn how to build a simple Android app. You'll do
the following:

*   [Set up your environment](#set-up-your-environment)
    *   [Install Bazel](#install-bazel)
    *   [Install Android Studio](#install-android-studio)
    *   [Get the sample project](#get-the-sample-project)
*   [Set up a workspace](#set-up-a-workspace)
    *   [Create a WORKSPACE file](#create-a-workspace-file)
    *   [Update the WORKSPACE file](#update-the-workspace-file)
*   [Review the source files](#review-the-source-files)
*   [Create a BUILD file](#create-a-build-file)
    *   [Add an android_library rule](#add-an-android_library-rule)
    *   [Add an android_binary rule](#add_an-android_binary-rule)
*   [Build the app](#build-the-app)
*   [Find the build outputs](#find-the-build-outputs)
*   [Run the app](#run-the-app)
*   [Review  your work](#review-your-work)

## Set up your environment

To get started, install Bazel and Android Studio, and get the sample project.

### Install Bazel

Follow the [installation instructions](../install.md) to install Bazel and
its dependencies.

### Install Android Studio

Download and install Android Studio as described in [Install Android Studio](https://developer.android.com/sdk/index.html).

The installer does not automatically set the `ANDROID_HOME` variable.
Set it to the location of the Android SDK, which defaults to `$HOME/Android/Sdk/`
.

For example:

`export ANDROID_HOME=$HOME/Android/Sdk/`

For convenience, add the above statement to your `~/.bashrc` file.

### Get the sample project

You also need to get the sample project for the tutorial from GitHub. The repo
has two branches: `source-only` and `master`. The `source-only` branch contains
the source files for the project only. You'll use the files in this branch in
this tutorial. The `master` branch contains both the source files and completed
Bazel `WORKSPACE` and `BUILD` files. You can use the files in this branch to
check your work when you've completed the tutorial steps.

Enter the following at the command line to get the files in the `source-only`
branch:

```bash
cd $HOME
git clone -b source-only https://github.com/bazelbuild/examples
```

The `git clone` command creates a directory named `$HOME/examples/`. This
directory contains several sample projects for Bazel. The project files for this
tutorial are in `$HOME/examples/tutorial/android`.

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

Note that Bazel itself doesn't make any requirements about how you organize
source files in your workspace. The sample source files in this tutorial are
organized according to conventions for the target platform.

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
```
This creates the empty `WORKSPACE` file.

### Update the WORKSPACE file

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
    name = "androidsdk"
)
```

This will use the Android SDK referenced by the `ANDROID_HOME` environment
variable, and automatically detect the highest API level and the latest version
of build tools installed within that location.

Alternatively, you can explicitly specify the location of the Android
SDK, the API level, and the version of build tools to use by including the
`path`,`api_level`, and `build_tools_version` attributes. You can specify any
subset of these attributes:

```python
android_sdk_repository(
    name = "androidsdk",
    path = "/path/to/Android/sdk",
    api_level = 25,
    build_tools_version = "26.0.1"
)
```

**Optional:** This is not required by this tutorial, but if you want to compile
native code into your Android app, you also need to download the
[Android NDK](https://developer.android.com/ndk/downloads/index.html) and
tell Bazel where to find it by adding the following rule to your `WORKSPACE`
file:

```python
android_ndk_repository(
    name = "androidndk"
)
```

`api_level` is the version of the Android API the SDK and the NDK target
(for example, 23 for Android 6.0 and 25 for Android 7.1). If not explicitly
set, `api_level` will default to the highest available API level for
`android_sdk_repository` and `android_ndk_repository`. It's not necessary to
set the API levels to the same value for the SDK and NDK.
[This web page](https://developer.android.com/ndk/guides/stable_apis.html)
contains a map from Android releases to NDK-supported API levels.

Similar to `android_sdk_repository`, the path to the Android NDK is inferred
from the `ANDROID_NDK_HOME` environment variable by default. The path can also
be explicitly specified with a `path` attribute on `android_ndk_repository`.

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

## Create a BUILD file

A [`BUILD` file](../build-ref.html#BUILD_files) is a text file that describes
the relationship between a set of build outputs -- for example, compiled
software libraries or executables -- and their dependencies. These dependencies
may be source files in your workspace or other build outputs. `BUILD` files are
written in the Bazel *build language*.

`BUILD` files are part of a concept in Bazel known as the *package hierarchy*.
The package hierarchy is a logical structure that overlays the directory
structure in your workspace. Each [package](../build-ref.html#packages) is a
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
vi $WORKSPACE/android/BUILD
```

### Add an android_library rule

A `BUILD` file contains several different types of instructions for Bazel. The
most important type is the [build rule](../build-ref.html#funcs), which tells
Bazel how to build an intermediate or final software output from a set of source
files or other dependencies.

Bazel provides two build rules, `android_library` and `android_binary`, that you
can use to build an Android app. For this tutorial, you'll first use the
[`android_library`](../be/android.html#android_library) rule to tell
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

The [`android_binary`](../be/android.html#android_binary) rule builds
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

## Build the app

You use the
[`bazel`](../user-manual.html) command-line tool to run builds, execute
unit tests and perform other operations in Bazel. This tool is located in the
`output` subdirectory of the location where you installed Bazel. During
[installation](../install.md), you probably added this location to your
path.

Before you build the sample app, make sure that your current working directory
is inside your Bazel workspace:

```bash
cd $WORKSPACE
```

Now, enter the following to build the sample app:

```bash
bazel build //android:android
```

The [`build`](../user-manual.html#build) subcommand instructs Bazel to
build the target that follows. The target is specified as the name of a build
rule inside a `BUILD` file, with along with the package path relative to
your workspace directory. Note that you can sometimes omit the package path
or target name, depending on your current working directory at the command
line and the name of the target. See [Labels](../build-ref.html#labels) in the
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
ls $WORKSPACE/bazel-bin/android
```

## Run the app

**NOTE:** The app launches standalone but requires a backend server in order to
produce output. See the README file in the sample project directory to find out
how to build the backend server.

You can now deploy the app to a connected Android device or emulator from the
command line using the
[`bazel mobile-install`](../user-manual.html#mobile-install)
command. This command uses the Android Debug Bridge (`adb`) to communicate with
the device. You must set up your device to use `adb` following the instructions
in
[Android Debug Bridge](http://developer.android.com/tools/help/adb.html) before
deployment. You can also choose to install the app on the Android emulator
included in Android Studio. Make sure the emulator is running before executing
the command below.

Enter the following:

```bash
bazel mobile-install //android:android
```

Note that the `mobile-install` subcommand also supports the
[`--incremental`](../user-manual.html#mobile-install)
flag that can be used to deploy only those parts of the app that have changed
since the last deployment.

## Review your work

In this tutorial, you used Bazel to build an Android app. To accomplish that,
you:

*   Set up your environment by installing Bazel and Android Studio, and
    downloading the sample project
*   Set up a Bazel [workspace](workspace.md) that contained the source code
    for the app and a `WORKSPACE` file that identifies the top level of the
    workspace directory
*   Updated the `WORKSPACE` file to contain references to the required
    external dependencies
*   Created a `BUILD` file
*   Ran Bazel to build the app
*   Deployed and ran the app on an Android emulator and device

The built app is located in the `$WORKSPACE/bazel-bin` directory.

Note that completed `WORKSPACE` and `BUILD` files for this tutorial are located
in the
[master branch](https://github.com/bazelbuild/examples/tree/master/tutorial)
of the GitHub repo. You can compare your work to the completed files for
additional help or troubleshooting.
