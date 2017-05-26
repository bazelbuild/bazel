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

\---

## FULL RELEASE NOTES:

### Incompatible changes:

- Bazel's Linux sandbox no longer mounts an empty tmpfs on /tmp,
  instead the existing /tmp is mounted read-write. If you prefer
  to have a tmpfs on /tmp for sandboxed actions for increased
  hermeticity, please use the flag --sandbox\_tmpfs\_path=/tmp.
- Converting artifacts to strings and printing them now return
  "File" instead of "Artifact" to be consistent with the type name.
- The return type of depset.to\_list() is now a list rather than a
  frozen list. (Modifying the list has no effect on the depset.)
- Bazel now prints logs in single lines to java.log
- --use\_dash, --dash\_url and --dash\_secret are removed.
- Remote repositories must define any remote repositories they
  themselves use (e.g., if @x//:foo depends on @y//:bar, @y must be
  defined
  in @x's WORKSPACE file).
- Remote repositories must define any remote repositories they
  themselves use (e.g., if @x//:foo depends on @y//:bar, @y must be
  defined
  in @x's WORKSPACE file).
- objc\_xcodeproj has been removed, use tulsi.bazel.build instead.

### New features:

- If grte\_top is a label, it can now follow non-configurable
  redirects.
- Optional coverage\_files attribute to cc\_toolchain
- "query --output=build" now includes select()s
- Raw LLVM profiles are now supported.

### Important changes:

- Automatically generate Proguard mapping when resource shrinking
  and Proguard are enabled.
- New rules in Bazel: proto\_library, java\_lite\_proto\_library,
  java\_proto\_library and cc\_proto\_library
- Activate the "dead\_strip" feature if objc binary stripping is
  enabled.
- More stable naming scheme for lambda classes in desugared android
  code
- Convert --use\_action\_cache to a regular option
- Per-architecture dSYM binaries are now propagated by
  apple\_binary's AppleDebugOutputsProvider.
- Avoid factory methods when desugaring stateless lambdas for
  Android
- desugar calls to Objects.requireNonNull(Object o) with
  o.getClass() for android
- Add an --copy\_bridges\_from\_classpath argument to android
  desugaring tool
- Change how desugar finds desugared classes to have it working on
  Windows
- Evaluation of commands on TargetsBelowDirectory patterns
  (e.g. //foo/...) matching packages that fail to load now report
  more
  detailed error messages in keep\_going mode.
- Allow to have several inputs and outputs
- Repository context's execute() function can print stdout/stderr
  while running. To enable, pass quiet=False.
- Bazel can now be built with a bundled version of the OpenJDK.
  This makes it possible to use Bazel on systems without a JDK, or
  where
  the installed JDK is too old.
- The --jobs flag now defaults to "auto", which causes Bazel to
  use a reasonable degree of parallelism based on the local
  machine's
  capacity.
- Bazel benchmark (perf.bazel.build) supports Java and Cpp targets.
- no factory methods generated for lambda expressions on android
- The Linux sandbox no longer changes the user to 'nobody' by
  default, instead the current user is used as is. The old behavior
  can be
  restored via the --sandbox\_fake\_username flag.
- /tmp and /dev/shm are now writable by default inside the
  Linux sandbox.
- Bazel can now use the process-wrapper + symlink tree based
  sandbox implementation in FreeBSD.
- turn on --experimental\_incremental\_dexing\_error\_on\_missed\_jars by
  default.
- All android\_binarys are now signed with both Apk Signature V1 and
  V2. See https://source.android.com/security/apksigning/v2.html
  for more details.
- Windows MSVC wrappers: Not filtering warning messages anymore,
  use --copt=-w and --host\_copt=-w to suppress them.
- A downloader bug was fixed that prevented RFC 7233 Range
  connection resumes from working with certain HTTP servers
- Introduces experimental android\_device rule for configuring and
  launching Android emulators.
- For boolean flags, setting them to false using --no\_<flag\_name>
  is deprecated. Use --no<flag\_name> without the underscore, or
  --<flag\_name>=false instead.
- Add --experimental\_android\_compress\_java\_resources flag to store
  java
  resources as compressed inside the APK.
- Removed --experimental\_use\_jack\_for\_dexing and libname.jack
  output of
  android\_library.
- blaze canonicalize-flags now takes a --show\_warnings flag
- Changing --invocation\_policy will no longer force a server
  restart.
- Bazel now supports Android NDK14.
- android\_binary multidex should now work without additional flags.
- Use action\_config in crosstool for static library archiving,
  remove ar\_flag.
- new option for bazel canonicalize-flags, --canonicalize\_policy
- Use action\_config in crosstool for static library archiving,
  remove ar\_flag.
- android\_library exports\_manifest now defaults to True.
- Fix select condition intersections.
- Adds a --override\_repository option that takes a repository
  name and path. This forces Bazel to use the directory at that path
  for the repository. Example usage:
  `--override_repository=foo=/home/user/gitroot/foo`.
- fix idempotency issue with desugaring lambdas in interface
  initializers for android
- --experimental\_android\_use\_singlejar\_for\_multidex is now a no-op
  and will eventually be removed.
- Every local\_repository now requires a WORKSPACE file.
- Remove jack and jill attributes of the android\_sdk rule.
- Add Skylark stubs needed to remove sysroot from CppConfiguration.
- Desugar try-with-resources so that this language feature is
  available
  to deveces with API level under 19.
- The flag --worker\_max\_retries was removed. The
  WorkerSpawnStrategy no longer retries execution of failed Spawns,
  the reason being that this just masks compiler bugs and isn't
  done for any other execution strategy either.
- Bazel will no longer gracefully restart workers that crashed /
  quit, instead this triggers a build failure.
- All java resources are now compressed in android\_binary APKs by
  default.
- All java resources are now compressed in android\_binary APKs by
  default.
- android\_ndk\_repository now creates a cc\_library
  (@androidndk//:cpufeatures) for the cpufeatures library that is
  bundled in the Android NDK. See
  https://developer.android.com/ndk/guides/cpu-features.html for
  more details.
- 'output\_groups' and 'instrumented\_files' cannot be specified in
  DefaultInfo.
- You can increase the CPU reservation for tests by adding a
  "cpu:<n>" (e.g. "cpu:4" for four cores) tag to their rule in a
  BUILD file. This can be used if tests would otherwise overwhelm
  your system if there's too much parallelism.
- Deprecate use\_singlejar\_for\_proguard\_libraryjars and force
  behavior to always on.
