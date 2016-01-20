---
layout: documentation
title: Tutorial - Set Up a Workspace
---

# Tutorial - Set Up a Workspace

A [workspace](/docs/build-ref.html#workspaces) is a directory that contains the
source files for one or more software projects, as well as a `WORKSPACE` file
and `BUILD` files that contain the instructions that Bazel uses to build
the software. It also contains symbolic links to output directories in the
Bazel home directory.

A workspace directory can be located anywhere on your filesystem. In this
tutorial, your workspace directory is `$HOME/examples/tutorial/`, which
contains the sample project files you cloned from the GitHub repo in the
previous step.

Note that Bazel itself doesn't make any requirements about how you organize
source files in your workspace. The sample source files in this tutorial are
organized according to common conventions for Android apps, iOS apps and App
Engine applications.

For your convenience, set the `$WORKSPACE` environment variable now to refer to
your workspace directory. At the command line, enter:

```bash
$ export WORKSPACE=$HOME/examples/tutorial
```

## Create a WORKSPACE file

Every workspace must have a text file named `WORKSPACE` located in the top-level
workspace directory. This file may be empty or it may contain references
to [external dependencies](/docs/external.html) required to build the
software.

For now, you'll create an empty `WORKSPACE` file, which simply serves to
identify the workspace directory. In later steps, you'll update the file to add
external dependency information.

Enter the following at the command line:

```bash
$ touch $WORKSPACE/WORKSPACE
```

This creates the empty `WORKSPACE` file.

## What's next

Now that you've set up your workspace, you can
[build the Android app](android-app.md).
