---
layout: documentation
title: Android and Bazel
---

# Android and Bazel

This page contains resources that help you use Bazel with Android projects. It
links to a tutorial, build rules, and other information specific to building
Android projects with Bazel.

## Getting started

The following resources will help you work with Bazel on Android projects:

*  [Tutorial: Building an Android app](tutorial/android-app.html). This tutorial
   is a good place to start learning about Bazel commands and concepts, and how
   to build Android apps with Bazel.
*  [Codelab: Building Android Apps with Bazel](https://codelabs.developers.google.com/codelabs/bazel-android-intro/index.html).
   This codelab explains how to build Android apps with Bazel.

## Features

Bazel has Android rules for building and testing Android apps, integrating with
the SDK/NDK, and creating emulator images. There are also Bazel plugins for
Android Studio and IntelliJ.

*  [Android rules](be/android.html). The Build Encyclopedia describes the rules
   you can use to build and test Android apps with Bazel.
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

## Further reading

*  Learn [How Android Builds Work in Bazel](https://blog.bazel.build/2018/02/14/how-android-builds-work-in-bazel.html).
