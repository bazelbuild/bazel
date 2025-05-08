Project: /_project.yaml
Book: /_book.yaml

# Bazel Tutorial: Build an Android App

{% include "_buttons.html" %}
**Note:** There are known limitations on using Bazel for building Android apps.
Visit the Github [team-Android hotlist](https://github.com/bazelbuild/bazel/issues?q=is%3Aissue+is%3Aopen+label%3Ateam-Android) to see the list of known issues. While the Bazel team and Open Source Software (OSS) contributors work actively to address known issues, users should be aware that Android Studio does not officially support Bazel projects.

This tutorial covers how to build a simple Android app using Bazel.

Bazel supports building Android apps using the
[Android rules](/reference/be/android).

This tutorial is intended for Windows, macOS and Linux users and does not
require experience with Bazel or Android app development. You do not need to
write any Android code in this tutorial.

## What you'll learn

In this tutorial you learn how to:

*   Set up your environment by installing Bazel and Android Studio, and
    downloading the sample project.
*   Set up a Bazel workspace that contains the source code
    for the app and a `MODULE.bazel` file that identifies the top level of the
    workspace directory.
*   Update the `MODULE.bazel` file to contain references to the required
    external dependencies, like the Android SDK.
*   Create a `BUILD` file.
*   Build the app with Bazel.
*   Deploy and run the app on an Android emulator or physical device.

## Before you begin

### Install Bazel

Before you begin the tutorial, install the following software:

* **Bazel.** To install, follow the [installation instructions](/install).
* **Android Studio.** To install, follow the steps to [download Android
  Studio](https://developer.android.com/sdk/index.html){: .external}.
  Execute the setup wizard to download the SDK and configure your environment.
* (Optional) **Git.** Use `git` to download the Android app project.

### Get the sample project

For the sample project, use a basic Android app project in
[Bazel's examples repository](https://github.com/bazelbuild/examples){: .external}.

This app has a single button that prints a greeting when clicked:

![Button greeting](/docs/images/android_tutorial_app.png "Tutorial app button greeting")

**Figure 1.** Android app button greeting.

Clone the repository with `git` (or [download the ZIP file
directly](https://github.com/bazelbuild/examples/archive/master.zip){: .external}):

```posix-terminal
git clone https://github.com/bazelbuild/examples
```

The sample project for this tutorial is in `examples/android/tutorial`. For
the rest of the tutorial, you will be executing commands in this directory.

### Review the source files

Take a look at the source files for the app.

```
.
├── README.md
└── src
    └── main
        ├── AndroidManifest.xml
        └── java
            └── com
                └── example
                    └── bazel
                        ├── AndroidManifest.xml
                        ├── Greeter.java
                        ├── MainActivity.java
                        └── res
                            ├── layout
                            │   └── activity_main.xml
                            └── values
                                ├── colors.xml
                                └── strings.xml
```

The key files and directories are:

| Name                    | Location                                                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| Android manifest files  | `src/main/AndroidManifest.xml` and `src/main/java/com/example/bazel/AndroidManifest.xml` |
| Android source files    | `src/main/java/com/example/bazel/MainActivity.java` and `Greeter.java`                   |
| Resource file directory | `src/main/java/com/example/bazel/res/`                                                   |


## Build with Bazel

### Set up the workspace

A [workspace](/concepts/build-ref#workspace) is a directory that contains the
source files for one or more software projects, and has a `MODULE.bazel` file at
its root.

The `MODULE.bazel` file may be empty or may contain references to [external
dependencies](/external/overview) required to build your project.

First, run the following command to create an empty `MODULE.bazel` file:

|          OS              |              Command                |
| ------------------------ | ----------------------------------- |
| Linux, macOS             | `touch MODULE.bazel`                   |
| Windows (Command Prompt) | `type nul > MODULE.bazel`              |
| Windows (PowerShell)     | `New-Item MODULE.bazel -ItemType file` |

### Running Bazel

You can now check if Bazel is running correctly with the command:

```posix-terminal
bazel info workspace
```

If Bazel prints the path of the current directory, you're good to go! If the
`MODULE.bazel` file does not exist, you may see an error message like:

```
ERROR: The 'info' command is only supported from within a workspace.
```

### Integrate with the Android SDK

Bazel needs to run the Android SDK
[build tools](https://developer.android.com/tools/revisions/build-tools.html){: .external}
to build the app. This means that you need to add some information to your
`MODULE.bazel` file so that Bazel knows where to find them.

Add the following line to your `MODULE.bazel` file:

```python
bazel_dep(name = "rules_android", version = "0.5.1")
```

This will use the Android SDK at the path referenced by the `ANDROID_HOME`
environment variable, and automatically detect the highest API level and the
latest version of build tools installed within that location.

You can set the `ANDROID_HOME` variable to the location of the Android SDK. Find
the path to the installed SDK using Android Studio's [SDK
Manager](https://developer.android.com/studio/intro/update#sdk-manager){: .external}.
Assuming the SDK is installed to default locations, you can use the following
commands to set the `ANDROID_HOME` variable:

|          OS              |               Command                               |
| ------------------------ | --------------------------------------------------- |
| Linux                    | `export ANDROID_HOME=$HOME/Android/Sdk/`            |
| macOS                    | `export ANDROID_HOME=$HOME/Library/Android/sdk`     |
| Windows (Command Prompt) | `set ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk`       |
| Windows (PowerShell)     | `$env:ANDROID_HOME="$env:LOCALAPPDATA\Android\Sdk"` |

The above commands set the variable only for the current shell session. To make
them permanent, run the following commands:

|          OS              |               Command                               |
| ------------------------ | --------------------------------------------------- |
| Linux                    | `echo "export ANDROID_HOME=$HOME/Android/Sdk/" >> ~/.bashrc`                                                                              |
| macOS                    | `echo "export ANDROID_HOME=$HOME/Library/Android/Sdk/" >> ~/.bashrc`                                                                              |
| Windows (Command Prompt) | `setx ANDROID_HOME "%LOCALAPPDATA%\Android\Sdk"`                                                                                          |
| Windows (PowerShell)     | `[System.Environment]::SetEnvironmentVariable('ANDROID_HOME', "$env:LOCALAPPDATA\Android\Sdk", [System.EnvironmentVariableTarget]::User)` |


**Optional:** If you want to compile native code into your Android app, you
also need to download the [Android
NDK](https://developer.android.com/ndk/downloads/index.html){: .external}
and use `rules_android_ndk` by adding the following line to your `MODULE.bazel` file:

```python
bazel_dep(name = "rules_android_ndk", version = "0.1.2")
```


For more information, read [Using the Android Native Development Kit with
Bazel](/docs/android-ndk).

It's not necessary to set the API levels to the same value for the SDK and NDK.
[This page](https://developer.android.com/ndk/guides/stable_apis.html){: .external}
contains a map from Android releases to NDK-supported API levels.

### Create a BUILD file

A [`BUILD` file](/concepts/build-files) describes the relationship
between a set of build outputs, like compiled Android resources from `aapt` or
class files from `javac`, and their dependencies. These dependencies may be
source files (Java, C++) in your workspace or other build outputs. `BUILD` files
are written in a language called **Starlark**.

`BUILD` files are part of a concept in Bazel known as the *package hierarchy*.
The package hierarchy is a logical structure that overlays the directory
structure in your workspace. Each [package](/concepts/build-ref#packages) is a
directory (and its subdirectories) that contains a related set of source files
and a `BUILD` file. The package also includes any subdirectories, excluding
those that contain their own `BUILD` file. The *package name* is the path to the
`BUILD` file relative to the `MODULE.bazel` file.

Note that Bazel's package hierarchy is conceptually different from the Java
package hierarchy of your Android App directory where the `BUILD` file is
located, although the directories may be organized identically.

For the simple Android app in this tutorial, the source files in `src/main/`
comprise a single Bazel package. A more complex project may have many nested
packages.

#### Add an android_library rule

A `BUILD` file contains several different types of declarations for Bazel. The
most important type is the
[build rule](/concepts/build-files#types-of-build-rules), which tells
Bazel how to build an intermediate or final software output from a set of source
files or other dependencies. Bazel provides two build rules,
[`android_library`](/reference/be/android#android_library) and
[`android_binary`](/reference/be/android#android_binary), that you can use to
build an Android app.

For this tutorial, you'll first use the
`android_library` rule to tell Bazel to build an [Android library
module](http://developer.android.com/tools/projects/index.html#LibraryProjects){: .external}
from the app source code and resource files. You'll then use the
`android_binary` rule to tell Bazel how to build the Android application package.

Create a new `BUILD` file in the `src/main/java/com/example/bazel` directory,
and declare a new `android_library` target:

`src/main/java/com/example/bazel/BUILD`:

```python
package(
    default_visibility = ["//src:__subpackages__"],
)

android_library(
    name = "greeter_activity",
    srcs = [
        "Greeter.java",
        "MainActivity.java",
    ],
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
)
```

The `android_library` build rule contains a set of attributes that specify the
information that Bazel needs to build a library module from the source files.
Note also that the name of the rule is `greeter_activity`. You'll reference the
rule using this name as a dependency in the `android_binary` rule.

#### Add an android_binary rule

The [`android_binary`](/reference/be/android#android_binary) rule builds
the Android application package (`.apk` file) for your app.

Create a new `BUILD` file in the `src/main/` directory,
and declare a new `android_binary` target:

`src/main/BUILD`:

```python
android_binary(
    name = "app",
    manifest = "AndroidManifest.xml",
    deps = ["//src/main/java/com/example/bazel:greeter_activity"],
)
```

Here, the `deps` attribute references the output of the `greeter_activity` rule
you added to the `BUILD` file above. This means that when Bazel builds the
output of this rule it checks first to see if the output of the
`greeter_activity` library rule has been built and is up-to-date. If not, Bazel
builds it and then uses that output to build the application package file.

Now, save and close the file.

### Build the app

Try building the app! Run the following command to build the
`android_binary` target:

```posix-terminal
bazel build //src/main:app
```

The [`build`](/docs/user-manual#build) subcommand instructs Bazel to build the
target that follows. The target is specified as the name of a build rule inside
a `BUILD` file, with along with the package path relative to your workspace
directory. For this example, the target is `app` and the package path is
`//src/main/`.

Note that you can sometimes omit the package path or target name, depending on
your current working directory at the command line and the name of the target.
For more details about target labels and paths, see [Labels](/concepts/labels).

Bazel will start to build the sample app. During the build process, its output
will appear similar to the following:

```bash
INFO: Analysed target //src/main:app (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //src/main:app up-to-date:
  bazel-bin/src/main/app_deploy.jar
  bazel-bin/src/main/app_unsigned.apk
  bazel-bin/src/main/app.apk
```

#### Locate the build outputs

Bazel puts the outputs of both intermediate and final build operations in a set
of per-user, per-workspace output directories. These directories are symlinked
from the following locations at the top-level of the project directory, where
the `MODULE.bazel` file is:

* `bazel-bin` stores binary executables and other runnable build outputs
* `bazel-genfiles` stores intermediary source files that are generated by
   Bazel rules
* `bazel-out` stores other types of build outputs

Bazel stores the Android `.apk` file generated using the `android_binary` rule
in the `bazel-bin/src/main` directory, where the subdirectory name `src/main` is
derived from the name of the Bazel package.

At a command prompt, list the contents of this directory and find the `app.apk`
file:

|          OS              |          Command         |
| ------------------------ | ------------------------ |
| Linux, macOS             | `ls bazel-bin/src/main`  |
| Windows (Command Prompt) | `dir bazel-bin\src\main` |
| Windows (PowerShell)     | `ls bazel-bin\src\main`  |


### Run the app

You can now deploy the app to a connected Android device or emulator from the
command line using the [`bazel
mobile-install`](/docs/user-manual#mobile-install) command. This command uses
the Android Debug Bridge (`adb`) to communicate with the device. You must set up
your device to use `adb` following the instructions in [Android Debug
Bridge](http://developer.android.com/tools/help/adb.html){: .external} before deployment. You
can also choose to install the app on the Android emulator included in Android
Studio. Make sure the emulator is running before executing the command below.

Enter the following:

```posix-terminal
bazel mobile-install //src/main:app
```

Next, find and launch the "Bazel Tutorial App":

![Bazel tutorial app](/docs/images/android_tutorial_before.png "Bazel tutorial app")

**Figure 2.** Bazel tutorial app.

**Congratulations! You have just installed your first Bazel-built Android app.**

Note that the `mobile-install` subcommand also supports the
[`--incremental`](/docs/user-manual#mobile-install) flag that can be used to
deploy only those parts of the app that have changed since the last deployment.

It also supports the `--start_app` flag to start the app immediately upon
installing it.

## Further reading

For more details, see these pages:

* Open issues on [GitHub](https://github.com/bazelbuild/bazel/issues)
* More information on [mobile-install](/docs/mobile-install)
* Integrate external dependencies like AppCompat, Guava and JUnit from Maven
  repositories using [rules_jvm_external](https://github.com/bazelbuild/rules_jvm_external){: .external}
* Run Robolectric tests with the [robolectric-bazel](https://github.com/robolectric/robolectric-bazel){: .external}
  integration.
* Testing your app with [Android instrumentation tests](/docs/android-instrumentation-test)
* Integrating C and C++ code into your Android app with the [NDK](/docs/android-ndk)
* See more Bazel example projects of:
  * [a Kotlin app](https://github.com/bazelbuild/rules_jvm_external/tree/master/examples/android_kotlin_app){: .external}
  * [Robolectric testing](https://github.com/bazelbuild/rules_jvm_external/tree/master/examples/android_local_test){: .external}
  * [Espresso testing](https://github.com/bazelbuild/rules_jvm_external/tree/master/examples/android_instrumentation_test){: .external}

Happy building!
