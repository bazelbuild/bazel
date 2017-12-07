---
layout: documentation
title: Android and Bazel
---

# Android and Bazel

This page contains resources that help you use Bazel with Android projects. It
links to a tutorial, build rules, and other information specific to building
Android projects with Bazel.

## Contents

- [Working with Bazel](#working-with-bazel)
- [Android and Skylark](#android-and-skylark)

## Working with Bazel

The following resources will help you work with Bazel on Android projects:

*  [Tutorial: Building an Android app](tutorial/android-app.html)
*  [Android rules](https://docs.bazel.build/versions/master/be/android.html)
*  [mobile-install for Android](mobile-install.html)
*  [Integration with Android Studio](ide.html)

## Android and Skylark

**Note**: Extending Bazel with Skylark is for advanced build and test scenarios.
You do not need to use Skylark when getting started with Bazel.

The following [Skylark](https://docs.bazel.build/versions/master/skylark/concepts.html)
modules, configuration fragments, and providers will help you extend Bazel's
capabilities when building your Android projects:

*  Modules:

   *  [`android_common`](skylark/lib/AndroidSkylarkApiProvider.html)
   *  [`AndroidSkylarkIdlInfo`](skylark/lib/AndroidSkylarkIdlInfo.html)

*  Configuration fragments:

   *  [`android`](skylark/lib/android.html)

*  Providers:

   *  [`android`](skylark/lib/AndroidSkylarkApiProvider.html)
