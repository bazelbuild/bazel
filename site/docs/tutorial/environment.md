---
layout: documentation
title: Tutorial - Set Up Your Environment
---

# Tutorial - Set Up Your Environment

The first step in this tutorial is to set up your environment.

Here, you'll do the following:

*   Install Bazel
*   Install Android Studio and the Android SDK
*   Install Xcode (OS X only)
*   Get the sample project from the GitHub repo

## Install Bazel

Follow the [installation instructions](/docs/install.md) to install Bazel and
its dependencies.

## Install the Android SDK tools

Do the following:

1.  Download and install the
    [Android SDK Tools](https://developer.android.com/sdk/index.html#Other).

2.  Run the Android SDK Manager and install the following packages:

    <table class="table table-condensed table-striped">
    <thead>
    <tr>
    <td>Package</td>
    <td>SDK directory</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Android SDK Platform Tools</td>
    <td><code>platform-tools</code></td>
    </tr>
    <tr>
    <td>Android SDK Build Tools</td>
    <td><code>build-tools</code></td>
    </tr>
    <tr>
    <td>Android SDK Platform</td>
    <td><code>platform</code></td>
    </tr>
    </tbody>
    </table>

    The SDK Manager is an executable named `android` located in the `tools`
    directory.

## Install Xcode (OS X only)

If you are following the steps in this tutorial on Mac OS X, download and
install [Xcode](https://developer.apple.com/xcode/downloads/). The Xcode
download contains the iOS libraries, Objective-C compiler other tools
required by Bazel to build the iOS app.

## Get the sample project

You also need to get the sample project for the tutorial from GitHub:

[https://github.com/bazelbuild/examples/](https://github.com/bazelbuild/examples/)

The GitHub repo has two branches: `source-only` and `master`. The `source-only`
branch contains the source files for the project only. You'll use the files in
this branch in this tutorial. The `master` branch contains both the source files
and completed Bazel `WORKSPACE` and `BUILD` files. You can use the files in this
branch to check your work when you've completed the tutorial steps.

Enter the following at the command line to get the files in the `source-only`
branch:

```bash
$ cd $HOME
$ git clone -b source-only https://github.com/bazelbuild/examples
```

The `git clone` command creates a directory named `$HOME/examples/`. This
directory contains several sample projects for Bazel. The project files for this
tutorial are in `$HOME/examples/tutorial`.

## What's next

Now that you have set up your environment, you can
[set up a Bazel workspace](workspace.md).
