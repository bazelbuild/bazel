---
layout: posts
title: Bazel 0.5.0 Released
---

We are delighted to announce the [0.5.0 release of
Bazel](https://github.com/bazelbuild/bazel/releases/tag/0.5.0) (follow the link
for the full release notes and list of changes).

This release simplifies Bazel installation on Windows and platforms where a JDK
is not available. It solidifies the Build Event Protocol and [Remote Execution
APIs](https://docs.google.com/document/d/1AaGk7fOPByEvpAbqeXIyE8HX_A3_axxNnvroblTZ_6s/edit).

**Known issue on MacOS**

Bazel release 0.5.0 contains a bug in the compiler detection on macOS which
requires Xcode and the iOS tooling to be installed
([corresponding issue #3063](https://github.com/bazelbuild/bazel/issues/3063)).
If you had Command Line Tools installed, you also need to switch to Xcode using
`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.

## Improvements from our roadmap

### Bundled JDK

As announced earlier, when using an install script, bazel now comes by default
bundled with JDK 8. This means fewer steps required to install Bazel.  Read more
about JDK 7 deprecation in [the related blog
post](https://bazel.build/blog/2017/04/21/JDK7-deprecation.html).

### Windows support: now in beta

Bazel on Windows is now easier to install: it is no longer linked with MSYS. A
following blog post will detail this further.  Bazel is now able to build Java,
C++ and Python on Windows.

### Build Event Protocol

The [Build Event
Protocol](https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/buildeventstream/proto/build_event_stream.proto)
is now available as an experimental option; it enables programmatic subscription
to Bazel's events (build started, action status, target completed, test
results…). Currently, the protocol can only be written to a file. A gRPC
transport is already in the works and will be added in the next minor release.
The API will be stabilized in 0.5.1.

### Coverage support for pure Java targets

Use bazel coverage //my:target to generate coverage information from a
java\_test.

## Other major changes since 0.4.0

### New rules

New rules in Bazel:
[proto\_library](https://bazel.build/versions/master/docs/be/protocol-buffer.html#proto_library),
[java\_lite\_proto\_library](https://bazel.build/versions/master/docs/be/java.html#java_lite_proto_library),
[java\_proto\_library](https://bazel.build/versions/master/docs/be/java.html#java_proto_library)
and
[cc\_proto\_library](https://bazel.build/versions/master/docs/be/c-cpp.html#cc_proto_library).

### New Apple rules

There is a new repository for building for Apple platforms:
[https://github.com/bazelbuild/rules\_apple](https://github.com/bazelbuild/rules_apple).
These rules replace the deprecated iOS/watchOS rules built into Bazel. By
rebuilding the rules from the ground up in Skylark and hosting them separately,
we can more quickly fix bugs and implement new Apple features and platform
versions as they become available.

### Android Support Improvements

-  Integration with the Android Support Repository libraries in
   android\_sdk\_repository.
-  Support for Java 8 in Android builds with --experimental\_desugar\_for\_android.
   See [Android Studio's
   documentation](https://developer.android.com/studio/preview/features/java8-support.html)
   for more details about Android's Java 8 language features.
-  Multidex is now fully supported via
   [android\_binary.multidex](https://bazel.build/versions/master/docs/be/android.html#android_binary.multidex).
-  android\_ndk\_repository now supports Android NDK 13 and NDK 14.
-  APKs are now signed with both APK Signature V1 and V2.
   See [Android
   documentation](https://source.android.com/security/apksigning/v2.html) for more
   details about APK Signature Scheme v2.

### Remote Execution API

We fixed a number of bugs in the Remote Execution implementation. The final RPC
API design has been sent to
[bazel-discuss@](https://groups.google.com/forum/#!forum/bazel-discuss) for
discussion (see [Design Document: Remote Execution
API](https://docs.google.com/document/d/1AaGk7fOPByEvpAbqeXIyE8HX_A3_axxNnvroblTZ_6s/edit#heading=h.ole76l21af90))
and it should be finalized in the 0.6.0 release. The final API should only be a
minor change compared to the implementation in this 0.5.0 release.

## Skylark

-  Declared Providers are now implemented and
   [documented](https://bazel.build/versions/master/docs/skylark/rules.html#providers).
   They enable more robust and clearly defined interfaces between different
   rules and aspects. We recommend using them for all rules and aspects.
-  The type formerly known as 'set' is now called 'depset'. Depsets make your
   rules perform much better, allowing rules memory consumption to scale
   linearly instead of quadratically with build graph size - make sure you have
   read the
   [documentation on depsets](https://bazel.build/versions/master/docs/skylark/depsets.html).

## Finally...

A big thank you to our community for your continued support.
Particular shout-outs to Peter Mounce for the [Chocolatey Windows
package](https://bazel.build/versions/master/docs/install-windows.html) and Yuki
Yugui Sonoda for maintaining [rules\_go](https://github.com/bazelbuild/rules_go)
(they both received an [open source peer
bonus](https://opensource.googleblog.com/2017/03/the-latest-round-of-google-open-source.html)
from Google).

Thank you all, keep the
[questions](http://stackoverflow.com/questions/tagged/bazel) and [bug
reports](https://github.com/bazelbuild/bazel/issues) coming!

See the full list of changes on [GitHub](https://github.com/bazelbuild/bazel/releases/tag/0.5.0).
