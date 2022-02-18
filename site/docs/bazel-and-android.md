---
layout: documentation
title: Android and Bazel
---

<div style="background-color: #EFCBCB; color: #AE2B2B;  border: 1px solid #AE2B2B; border-radius: 5px; border-left: 10px solid #AE2B2B; padding: 0.5em;">
<b>IMPORTANT:</b> The Bazel docs have moved! Please update your bookmark to <a href="https://bazel.build/docs/bazel-and-android" style="color: #0000EE;">https://bazel.build/docs/bazel-and-android</a>
<p/>
You can <a href="https://blog.bazel.build/2022/02/17/Launching-new-Bazel-site.html" style="color: #0000EE;">read about</a> the migration, and let us <a href="https://forms.gle/onkAkr2ZwBmcbWXj7" style="color: #0000EE;">know what you think</a>.
</div>


# Android and Bazel

This page contains resources that help you use Bazel with Android projects. It
links to a tutorial, build rules, and other information specific to building
Android projects with Bazel.

## Getting started

The following resources will help you work with Bazel on Android projects:

*  [Tutorial: Building an Android app](tutorial/android-app.html). This tutorial
   is a good place to start learning about Bazel commands and concepts, and how
   to build Android apps with Bazel.
*  [Codelab: Building Android Apps with Bazel](https://developer.android.com/codelabs/bazel-android-intro#0).
   This codelab explains how to build Android apps with Bazel.

## Features

Bazel has Android rules for building and testing Android apps, integrating with
the SDK/NDK, and creating emulator images. There are also Bazel plugins for
Android Studio and IntelliJ.

*  [Android rules](be/android.html). The Build Encyclopedia describes the rules
   for building and testing Android apps with Bazel.
*  [Integration with Android Studio](ide.html). Bazel is compatible with
   Android Studio using the [Android Studio with Bazel](https://ij.bazel.build/)
   plugin.
*  [`mobile-install` for Android](mobile-install.html). Bazel's `mobile-install`
   feature provides automated build-and-deploy functionality for building and
   testing Android apps directly on Android devices and emulators.
*  [Android instrumentation testing](android-instrumentation-test.html) on
   emulators and devices.
*  [Android NDK integration](android-ndk.html). Bazel supports compiling to
   native code through direct NDK integration and the C++ rules.
*  [Android build performance](android-build-performance.html). This page
   provides information on optimizing build performance for Android apps.

## Further reading

*  Integrating with dependencies from Google Maven and Maven Central with [rules_jvm_external](https://github.com/bazelbuild/rules_jvm_external).
*  Learn [How Android Builds Work in Bazel](https://blog.bazel.build/2018/02/14/how-android-builds-work-in-bazel.html).
