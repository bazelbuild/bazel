---
layout: documentation
title: Tutorial - Set Up Your Environment
---

# Tutorial - Set Up Your Environment

The first step in this tutorial is to set up your environment.

Here, you'll do the following:

*   Install Bazel
*   Install Android Studio
*   Install Xcode (macOS only)
*   Get the sample project from the GitHub repo

## Install Bazel

Follow the [installation instructions](../install.md) to install Bazel and
its dependencies.

## Install Android Studio

Download and install Android Studio as described in [Install Android Studio](https://developer.android.com/sdk/index.html).

The installer does not automatically set the `ANDROID_HOME` variable.
Set it to the location of the Android SDK, which defaults to `$HOME/Android/Sdk/`
.

For example:

`export ANDROID_HOME=$HOME/Android/Sdk/`

For convenience, add the above statement to your `~/.bashrc` file.

## Install Xcode (macOS only)

If you are following the steps in this tutorial on macOS, download and
install [Xcode](https://developer.apple.com/xcode/downloads/). The Xcode
download contains the iOS libraries, the Objective-C compiler, and other tools
required by Bazel to build iOS apps.

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
cd $HOME
git clone -b source-only https://github.com/bazelbuild/examples
```

The `git clone` command creates a directory named `$HOME/examples/`. This
directory contains several sample projects for Bazel. The project files for this
tutorial are in `$HOME/examples/tutorial`.

## What's next

Now that you have set up your environment, you can
[set up a Bazel workspace](workspace.md).
