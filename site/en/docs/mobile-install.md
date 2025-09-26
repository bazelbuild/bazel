Project: /_project.yaml
Book: /_book.yaml

# bazel mobile-install

{% include "_buttons.html" %}
{# disableFinding(LINK_EXTERNAL) #}

<p class="lead">Fast iterative development for Android</p>

This page describes how `bazel mobile-install` makes iterative development
for Android much faster. It describes the benefits of this approach versus the
drawbacks of separate build and install steps.

## Summary {:#summary}

To install small changes to an Android app very quickly, do the following:

 1. Find the `android_binary` rule of the app you want to install.
 2. Connect your device to `adb`.
 3. Run `bazel mobile-install :your_target`. App startup will be a little
    slower than usual.
 4. Edit the code or Android resources.
 5. Run `bazel mobile-install :your_target`.
 6. Enjoy a fast and minimal incremental installation!

Some command line options to Bazel that may be useful:

 - `--adb` tells Bazel which adb binary to use
 - `--adb_arg` can be used to add extra arguments to the command line of `adb`.
   One useful application of this is to select which device you want to install
   to if you have multiple devices connected to your workstation:
   `bazel mobile-install :your_target -- --adb_arg=-s --adb_arg=<SERIAL>`

When in doubt, look at the
[example](https://github.com/bazelbuild/rules_android/tree/main/examples/basicapp){: .external},
contact us on [Google Groups](https://groups.google.com/forum/#!forum/bazel-discuss){: .external},
or [file a GitHub issue](https://github.com/bazelbuild/rules_android/issues){: .external}

## Introduction {:#introduction}

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

## Problems with traditional app installation {:#problems-app-install}

Building an Android app has some issues, including:

- Dexing. By default, the Dexer tool (historically `dx`, now `d8` or `r8`)
is invoked exactly once in the build and it does not know how to reuse work from
previous builds: it dexes every method again, even though only one method was
changed.

- Uploading data to the device. adb does not use the full bandwidth of a USB 2.0
connection, and larger apps can take a lot of time to upload. The entire app is
uploaded, even if only small parts have changed, for example, a resource or a
single method, so this can be a major bottleneck.

- Compilation to native code. Android L introduced ART, a new Android runtime,
which compiles apps ahead-of-time rather than compiling them just-in-time like
Dalvik. This makes apps much faster at the cost of longer installation
time. This is a good tradeoff for users because they typically install an app
once and use it many times, but results in slower development where an app is
installed many times and each version is run at most a handful of times.

## The approach of `bazel mobile-install` {:#approach-mobile-install}

`bazel mobile-install `makes the following improvements:

 - Sharded desugaring and dexing. After building the app's Java code, Bazel
   shards the class files into approximately equal-sized parts and invokes `d8`
   separately on them. `d8` is not invoked on shards that did not change since
   the last build. These shards are then compiled into separate sharded APKs.

 - Incremental file transfer. Android resources, .dex files, and native
   libraries are removed from the main .apk and are stored in under a separate
   mobile-install directory. This makes it possible to update code and Android
   resources independently without reinstalling the whole app. Thus,
   transferring the files takes less time and only the .dex files that have
   changed are recompiled on-device.

 - Sharded installation. Mobile-install uses Android Studio's
   [`apkdeployer`](https://maven.google.com/web/index.html?q=deployer#com.android.tools.apkdeployer:apkdeployer){: .external}
   tool to combine sharded APKs on the connected device and provide a cohesive
   experience.

### Sharded Dexing {:#sharded-dexing}

Sharded dexing is reasonably straightforward: once the .jar files are built, a
[tool](https://github.com/bazelbuild/rules_android/blob/main/src/tools/java/com/google/devtools/build/android/ziputils/DexMapper.java){: .external}
shards them into separate .jar files of approximately equal size, then invokes
`d8` on those that were changed since the previous build. The logic that
determines which shards to dex is not specific to Android: it just uses the
general change pruning algorithm of Bazel.

The first version of the sharding algorithm simply ordered the .class files
alphabetically, then cut the list up into equal-sized parts, but this proved to
be suboptimal: if a class was added or removed (even a nested or an anonymous
one), it would cause all the classes alphabetically after it to shift by one,
resulting in dexing those shards again. Thus, it was decided to shard Java
packages rather than individual classes. Of course, this still results in
dexing many shards if a new package is added or removed, but that is much less
frequent than adding or removing a single class.

The number of shards is controlled by command-line configuration, using the
`--define=num_dex_shards=N` flag. In an ideal world, Bazel would
automatically determine how many shards are best, but Bazel currently must know
the set of actions (for example, commands to be executed during the build) before
executing any of them, so it cannot determine the optimal number of shards
because it doesn't know how many Java classes there will eventually be in the
app. Generally speaking, the more shards, the faster the build and the
installation will be, but the slower app startup becomes, because the dynamic
linker has to do more work. The sweet spot is usually between 10 and 50 shards.

### Incremental deployment {:#incremental-deployment}

Incremental APK shard transfer and installation is now handled by the
`apkdeployer` utility described in ["The approach of mobile-install"](#approach-mobile-install).
Whereas earlier (native) versions of mobile-install required manually tracking
first-time installations and selectively apply the `--incremental`
flag on subsequent installation, the most recent version in [`rules_android`](https://github.com/bazelbuild/rules_android/tree/main/mobile_install){: .external}
has been greatly simplified. The same mobile-install
invocation can be used regardless of how many times the app has been installed
or reinstalled.

At a high level, the `apkdeployer` tool is a wrapper around various `adb`
sub-commands. The main entrypoint logic can be found in the
[`com.android.tools.deployer.Deployer`](https://cs.android.com/android-studio/platform/tools/base/+/mirror-goog-studio-main:deploy/deployer/src/main/java/com/android/tools/deployer/Deployer.java){: .external}
class, with other utility classes colocated in the same package.
The `Deployer` class ingests, among other things, a list of paths to split
APKs and a protobuf with information about the installation, and leverages
deployment features for [Android app bundles](https://developer.android.com/guide/app-bundle){: .external}
in order to create an install session and incrementally deploy app splits.
See the [`ApkPreInstaller`](https://cs.android.com/android-studio/platform/tools/base/+/mirror-goog-studio-main:deploy/deployer/src/main/java/com/android/tools/deployer/ApkPreInstaller.java){: .external}
and [`ApkInstaller`](https://cs.android.com/android-studio/platform/tools/base/+/mirror-goog-studio-main:deploy/deployer/src/main/java/com/android/tools/deployer/ApkInstaller.java){: .external}
classes for implementation details.

## Results {:#results}

### Performance {:#performance}

In general, `bazel mobile-install` results in a 4x to 10x speedup of building
and installing large apps after a small change.

The following numbers were computed for a few Google products:

<img src="/docs/images/mobile-install-performance.svg"/>

This, of course, depends on the nature of the change: recompilation after
changing a base library takes more time.

### Limitations {:#limitations}

The tricks the stub application plays don't work in every case.
The following cases highlight where it does not work as expected:

 - Mobile-install is only supported via the Starlark rules of `rules_android`.
   See the ["brief history of mobile-install"](#mobile-install-history) for
   more detail.

 - Only devices running ART are supported. Mobile-install uses API and runtime features
   that only exist on devices running ART, not Dalvik. Any Android runtime more
   recent than Android L (API 21+) should be compatible.

 - Bazel itself must be run with a tool Java runtime _and_ language version
   of 17 or higher.

 - Bazel versions prior to 8.4.0 must specify some additional flags for
   mobile-install. See [the Bazel Android tutorial](/start/android-app). These
   flags inform Bazel where the Starlark mobile-install aspect is and which
   rules are supported.

### A brief history of mobile-install {:#mobile-install-history}
Earlier Bazel versions _natively_ included built-in build and test rules for
popular languages and ecosystems such as C++, Java, and Android. These rules
were therefore referred to as _native_ rules. Bazel 8 (released in 2024) removed
support for these rules because many of them had been migrated to the
[Starlark](/rules/language) language. See the ["Bazel 8.0 LTS blog post"](https://blog.bazel.build/2024/12/09/bazel-8-release.html){: .external}
for more details.

The legacy native Android rules also supported a legacy _native_ version of
mobile-install functionality. This is referred to as "mobile-install v1" or
"native mobile-install" now. This functionality was deleted in Bazel 8, along
with the built-in Android rules.

Now, all mobile-install functionality, as well as all Android build and test
rules, are implemented in Starlark and reside in the `rules_android` GitHub
repository. The latest version is known as "mobile-install v3" or "MIv3".

_Naming note_: There was a "mobile-install **v2**" available only internally
at Google at one point, but this was never published externally, and only v3
continues to be used for both Google-internal and OSS rules_android deployment.


