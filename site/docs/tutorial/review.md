---
layout: documentation
title: Tutorial - Review
---

# Tutorial - Review

In this tutorial, you used Bazel to build an [Android app](android-app.md),
an [iOS app](ios-app.md) and a [backend server](backend-server.md) that runs on
Google App Engine.

To build these software outputs, you:

*   Set up a Bazel [workspace](workspace.md) that contained the source code
    for the components and a `WORKSPACE` that identifies the top level of the
    workspace directory
*   Created a `BUILD` file for each component
*   Updated the `WORKSPACE` file to contain references to the required
    external dependencies
*   Ran Bazel to build the software components

The built mobile apps and backend server application files are located in the
`$WORKSPACE/bazel-bin` directory.

Note that completed `WORKSPACE` and `BUILD` files for this tutorial are located
in the
[master branch](https://github.com/bazelbuild/examples/tree/master/tutorial)
of the GitHub repo. You can compare your work to the completed files for
additional help or troubleshooting.
