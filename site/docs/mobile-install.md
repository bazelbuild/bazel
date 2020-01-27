---
layout: documentation
title: mobile-install
---

# bazel mobile-install

<p class="lead">Fast iterative development for Android</p>

## TL;DR

To install small changes to an Android app very quickly, do the following:

 1. Find the `android_binary` rule of the app you want to install.
 2. Disable Proguard by removing the `proguard_specs` attribute.
 3. Set the `multidex` attribute to `native`.
 4. Set the `dex_shards` attribute to `10`.
 5. Connect your device running ART (not Dalvik) over USB and enable USB
    debugging on it.
 6. Run `bazel mobile-install :your_target`. App startup will be a little
    slower than usual.
 7. Edit the code or Android resources.
 8. Run `bazel mobile-install --incremental :your_target`.
 9. Enjoy not having to wait a lot.

Some command line options to Bazel that may be useful:

 - `--adb` tells Bazel which adb binary to use
 - `--adb_arg` can be used to  add extra arguments to the command line of `adb`.
   One useful application of this is to select which device you want to install
   to if you have multiple devices connected to your workstation:
   `bazel mobile-install --adb_arg=-s --adb_arg=<SERIAL> :your_target`
 - `--start_app` automatically starts the app

When in doubt, look at the
[example](https://github.com/bazelbuild/bazel/tree/master/examples/android)
or [contact us](https://groups.google.com/forum/#!forum/bazel-discuss).

## Introduction

One of the most important attributes of a developer's toolchain is speed: there
is a world of difference between changing the code and seeing it run within a
second and having to wait minutes, sometimes hours, before you get any feedback
on whether your changes do what you expect them to.

Unfortunately, the traditional Android toolchain for building an .apk entails
many monolithic, sequential steps and all of these have to be done in order to
build an Android app. At Google, waiting five minutes to build a single-line
change was not unusual on larger projects like Google Maps.

`bazel mobile-install` makes iterative development for Android much faster by
using a combination of change pruning, work sharding, and clever manipulation of
Android internals, all without changing any of your app's code.

## Problems with traditional app installation

We identified the following bottlenecks of building an Android app:

- Dexing. By default, "dx" is invoked exactly once in the build and it does not
know how to reuse work from previous builds: it dexes every method again, even
though only one method was changed.

- Uploading data to the device. adb does not use the full bandwidth of a USB 2.0
connection, and larger apps can take a lot of time to upload.  The entire app is
uploaded, even if only small parts have changed, for example, a resource or a
single method, so this can be a major bottleneck.

- Compilation to native code. Android L introduced ART, a new Android runtime,
which compiles apps ahead-of-time rather than compiling them just-in-time like
Dalvik. This makes apps much faster at the cost of longer installation
time. This is a good tradeoff for users because they typically install an app
once and use it many times, but results in slower development where an app is
installed many times and each version is run at most a handful of times.

## The approach of `bazel mobile-install`

`bazel mobile-install `makes the following improvements:

 - Sharded dexing. After building the app's Java code, Bazel shards the class
   files into approximately equal-sized parts and invokes `dx` separately on
   them. `dx` is not invoked on shards that did not change since the last build.

 - Incremental file transfer. Android resources, .dex files, and native
   libraries are removed from the main .apk and are stored in under a separate
   mobile-install directory. This makes it possible to update code and Android
   resources independently without reinstalling the whole app. Thus,
   transferring the files takes less time and only the .dex files that have
   changed are recompiled on-device.

 - Loading parts of the app from outside the .apk. A tiny stub application is
   put into the .apk that loads Android resources, Java code and native code
   from the on-device mobile-install directory, then transfers control to the
   actual app. This is all transparent to the app, except in a few corner cases
   described below.

### Sharded Dexing

Sharded dexing is reasonably straightforward: once the .jar files are built, a
[tool](https://github.com/bazelbuild/bazel/blob/master/src/tools/android/java/com/google/devtools/build/android/ziputils/DexMapper.java)
shards them into separate .jar files of approximately equal size, then invokes
`dx` on those that were changed since the previous build. The logic that
determines which shards to dex is not specific to Android: it just uses the
general change pruning algorithm of Bazel.

The first version of the sharding algorithm simply ordered the .class files
alphabetically, then cut the list up into equal-sized parts, but this proved to
be suboptimal: if a class was added or removed (even a nested or an anonymous
one), it would cause all the classes alphabetically after it to shift by one,
resulting in dexing those shards again. Thus, we settled upon sharding not
individual classes, but Java packages instead. Of course, this still results in
dexing many shards if a new package is added or removed, but that is much less
frequent than adding or removing a single class.

The number of shards is controlled by the BUILD file (using the
`android_binary.dex_shards` attribute). In an ideal world, Bazel would
automatically determine how many shards are best, but Bazel currently must know
the set of actions (i.e. commands to be executed during the build) before
executing any of them, so it cannot determine the optimal number of shards
because it doesn't know how many Java classes there will eventually be in the
app. Generally speaking, the more shards, the faster the build and the
installation will be, but the slower app startup becomes, because the dynamic
linker has to do more work. The sweet spot is usually between 10 and 50 shards.

### Incremental File Transfer

After building the app, the next step is to install it, preferably with the
least effort possible. Installation consists of the following steps:

 1. Installing the .apk (i.e. `adb install`)
 2. Uploading the .dex files, Android resources, and native libraries to the
    mobile-install directory

There is not much incrementality in the first step: the app is either installed
or not. Bazel currently relies on the user to indicate if it should do this step
through the `--incremental` command line option because it cannot determine in
all cases if it is necessary.

In the second step, the app's files from the build are compared to an on-device
manifest file that lists which app files are on the device and their
checksums. Any new files are uploaded to the device, any files that have changed
are updated, and any files that have been removed are deleted from the
device. If the manifest is not present, it is assumed that every file needs to
be uploaded.

Note that it is possible to fool the incremental installation algorithm by
changing a file on the device, but not its checksum in the manifest. We could
have safeguarded against this by computing the checksum of the files on the
device, but this was deemed to be not worth the increase in installation time.

### The Stub Application

The stub application is where the magic to load the dexes, native code and
Android resources from the on-device `mobile-install` directory happens.

The actual loading is implemented by subclassing `BaseDexClassLoader` and is a
reasonably well-documented technique. This happens before any of the app's
classes are loaded, so that any application classes that are in the apk can be
placed in the on-device `mobile-install` directory so that they can be updated
without `adb install`.

This needs to happen before any of the
classes of the app are loaded, so that no application class needs to be in the
.apk which would mean that changes to those classes would require a full
re-install.

This is accomplished by replacing the `Application` class specified in
`AndroidManifest.xml` with the
[stub application](https://github.com/bazelbuild/bazel/blob/master/src/tools/android/java/com/google/devtools/build/android/incrementaldeployment/StubApplication.java). This
takes control when the app is started, and tweaks the class loader and the
resource manager appropriately at the earliest moment (its constructor) using
Java reflection on the internals of the Android framework.

Another thing the stub application does is to copy the native libraries
installed by mobile-install to another location. This is necessary because the
dynamic linker needs the `X` bit to be set on the files, which is not possible to
do for any location accessible by a non-root `adb`.

Once all these things are done, the stub application then instantiates the
actual `Application` class, changing all references to itself to the actual
application within the Android framework.

## Results

### Performance

In general, `bazel mobile-install` results in a 4x to 10x speedup of building
and installing large apps after a small change. We computed the following
numbers for a few Google products:

<img src="/assets/mobile-install-performance.svg"/>

This, of course, depends on the nature of the change: recompilation after
changing a base library takes more time.

### Limitations

The tricks the stub application plays don't work in every case. We have
identified the following cases where it does not work as expected:

 - When `Context` is cast to the `Application` class in
   `ContentProvider#onCreate()`. This method is called during application
   startup before we have a chance to replace the instance of the `Application`
   class, therefore, `ContentProvider` will still reference the stub application
   instead of the real one. Arguably, this is not a bug since you are not
   supposed to downcast `Context` like this, but this seems to happen in a few
   apps at Google.

 - Resources installed by `bazel mobile-install` are only available from within
   the app. If resources are accessed by other apps via
   `PackageManager#getApplicationResources()`, these resources will be from the
   last non-incremental install.

 - Devices that aren't running ART. While the stub application works well on
   Froyo and later, Dalvik has a bug that makes it think that the app is
   incorrect if its code is distributed over multiple .dex files in certain
   cases, for example, when Java annotations are used in a
   [specific](https://code.google.com/p/android/issues/detail?id=78144) way. As
   long as your app doesn't tickle these bugs, it should work with Dalvik, too
   (note, however, that support for old Android versions isn't exactly our
   focus)
