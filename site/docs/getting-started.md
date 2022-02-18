---
layout: documentation
title: Getting started
category: getting-started
---

<div style="background-color: #EFCBCB; color: #AE2B2B;  border: 1px solid #AE2B2B; border-radius: 5px; border-left: 10px solid #AE2B2B; padding: 0.5em;">
<b>IMPORTANT:</b> The Bazel docs have moved! Please update your bookmark to <a href="https://bazel.build/start/getting-started" style="color: #0000EE;">https://bazel.build/start/getting-started</a>
<p/>
You can <a href="https://blog.bazel.build/2022/02/17/Launching-new-Bazel-site.html" style="color: #0000EE;">read about</a> the migration, and let us <a href="https://forms.gle/onkAkr2ZwBmcbWXj7" style="color: #0000EE;">know what you think</a>.
</div>


# Getting Started with Bazel

This page contains resources that help you get started with Bazel, including
installation steps and container information. It also provides links to
tutorials and migration guides.

If you have not already done so, first read the [Bazel Overview](bazel-overview.html).

## Installation

To install Bazel, see [Installing Bazel](install.html).
If you use Windows, please read also [Using Bazel on Windows](windows.html).

You might also want to [integrate Bazel with your IDE](ide.html).

## Bazel container

To try out Bazel inside a [Docker](https://www.docker.com/) container, check out
our public Ubuntu Linux (16.04) based Bazel container in
[Google Cloud Marketplace](https://console.cloud.google.com/marketplace/details/google/bazel).

To get started with the Bazel container, check out [Getting started with Bazel Docker Container](bazel-container.html).

## Tutorials

To get hands-on with Bazel and understand its core concepts, complete a
tutorial:

*   [Tutorial: Build a C++ Project](tutorial/cpp.html)

*   [Tutorial: Build a Java Project](tutorial/java.html)

*   [Tutorial: Build an Android Application](tutorial/android-app.html)

*   [Tutorial: Build an iOS Application](tutorial/ios-app.html)

If you find yourself unsure of how Workspace, Packages, Targets and Rules
relate to each other, jump to the [Bazel Concepts](build-ref.html) page.

Once you are familiar with the basics, you can try the rules for
[other languages](rules.html).

## Migration

To learn how to migrate your project to Bazel, see the appropriate migration
guide:

*   [Migrating from Maven to Bazel](migrate-maven.html)

*   [Migrating from Xcode to Bazel](migrate-xcode.html)

## Reference

To further explore Bazel, refer to the following resources:

*   [Bazel Concepts and Terminology](build-ref.html)

*   [Bazel User Manual](user-manual.html)

*   [Rules](rules.html) for many languages
