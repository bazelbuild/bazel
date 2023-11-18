Project: /_project.yaml
Book: /_book.yaml

# Using the Android Native Development Kit with Bazel

{% include "_buttons.html" %}

_If you're new to Bazel, please start with the [Building Android with
Bazel](/start/android-app ) tutorial._

## Overview {:#overview}

Bazel can run in many different build configurations, including several that use
the Android Native Development Kit (NDK) toolchain. This means that normal
`cc_library` and `cc_binary` rules can be compiled for Android directly within
Bazel. Bazel accomplishes this by using the `android_ndk_repository` repository
rule.

## Prerequisites {:#prerequisites}

Please ensure that you have installed the Android SDK and NDK.

To set up the SDK and NDK, add the following snippet to your `WORKSPACE`:

```python
android_sdk_repository(
    name = "androidsdk", # Required. Name *must* be "androidsdk".
    path = "/path/to/sdk", # Optional. Can be omitted if `ANDROID_HOME` environment variable is set.
)

android_ndk_repository(
    name = "androidndk", # Required. Name *must* be "androidndk".
    path = "/path/to/ndk", # Optional. Can be omitted if `ANDROID_NDK_HOME` environment variable is set.
)
```

For more information about the `android_ndk_repository` rule, see the [Build
Encyclopedia entry](/reference/be/android#android_ndk_repository).

If you're using a recent version of the Android NDK (r22 and beyond), use the
Starlark implementation of `android_ndk_repository`.
Follow the instructions in
[its README](https://github.com/bazelbuild/rules_android_ndk).

## Quick start {:#quick-start}

To build C++ for Android, simply add `cc_library` dependencies to your
`android_binary` or `android_library` rules.

For example, given the following `BUILD` file for an Android app:

```python
# In <project>/app/src/main/BUILD.bazel

cc_library(
    name = "jni_lib",
    srcs = ["cpp/native-lib.cpp"],
)

android_library(
    name = "lib",
    srcs = ["java/com/example/android/bazel/MainActivity.java"],
    resource_files = glob(["res/**/*"]),
    custom_package = "com.example.android.bazel",
    manifest = "LibraryManifest.xml",
    deps = [":jni_lib"],
)

android_binary(
    name = "app",
    deps = [":lib"],
    manifest = "AndroidManifest.xml",
)
```

This `BUILD` file results in the following target graph:

![Example results](/docs/images/android_ndk.png "Build graph results")

**Figure 1.** Build graph of Android project with cc_library dependencies.

To build the app, simply run:

```posix-terminal
bazel build //app/src/main:app
```

The `bazel build` command compiles the Java files, Android resource files, and
`cc_library` rules, and packages everything into an APK:

```posix-terminal
$ zipinfo -1 bazel-bin/app/src/main/app.apk
nativedeps
lib/armeabi-v7a/libapp.so
classes.dex
AndroidManifest.xml
...
res/...
...
META-INF/CERT.SF
META-INF/CERT.RSA
META-INF/MANIFEST.MF
```

Bazel compiles all of the cc_libraries into a single shared object (`.so`) file,
targeted for the `armeabi-v7a` ABI by default. To change this or build for
multiple ABIs at the same time, see the section on [configuring the target
ABI](#configuring-target-abi).

## Example setup {:#example-setup}

This example is available in the [Bazel examples
repository](https://github.com/bazelbuild/examples/tree/master/android/ndk){: .external}.

In the `BUILD.bazel` file, three targets are defined with the `android_binary`,
`android_library`, and `cc_library` rules.

The `android_binary` top-level target builds the APK.

The `cc_library` target contains a single C++ source file with a JNI function
implementation:

```c++
#include <jni.h>
#include <string>

extern "C"
JNIEXPORT jstring

JNICALL
Java_com_example_android_bazel_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
```

The `android_library` target specifies the Java sources, resource files, and the
dependency on a `cc_library` target. For this example, `MainActivity.java` loads
the shared object file `libapp.so`, and defines the method signature for the JNI
function:

```java
public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("app");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
       // ...
    }

    public native String stringFromJNI();

}
```

Note: The name of the native library is derived from the name of the top
level `android_binary` target. In this example, it is `app`.

## Configuring the STL {:#configuring-stl}

To configure the C++ STL, use the flag `--android_crosstool_top`.

```posix-terminal
bazel build //:app --android_crosstool_top={{ "<var>" }}target label{{ "</var>" }}
```

The available STLs in `@androidndk` are:

| STL     | Target label                            |
|---------|-----------------------------------------|
| STLport | `@androidndk//:toolchain-stlport`       |
| libc++  | `@androidndk//:toolchain-libcpp`        |
| gnustl  | `@androidndk//:toolchain-gnu-libstdcpp` |

For r16 and below, the default STL is `gnustl`. For r17 and above, it is
`libc++`. For convenience, the target `@androidndk//:default_crosstool` is
aliased to the respective default STLs.

Please note that from r18 onwards, [STLport and gnustl will be
removed](https://android.googlesource.com/platform/ndk/+/master/docs/Roadmap.md#ndk-r18){: .external},
making `libc++` the only STL in the NDK.

See the [NDK
documentation](https://developer.android.com/ndk/guides/cpp-support){: .external}
for more information on these STLs.

## Configuring the target ABI {:#configuring-target-abi}

To configure the target ABI, use the `--fat_apk_cpu` flag as follows:

```posix-terminal
bazel build //:app --fat_apk_cpu={{ "<var>" }}comma-separated list of ABIs{{ "</var>" }}
```

By default, Bazel builds native Android code for `armeabi-v7a`. To build for x86
(such as for emulators), pass `--fat_apk_cpu=x86`. To create a fat APK for multiple
architectures, you can specify multiple CPUs: `--fat_apk_cpu=armeabi-v7a,x86`.

If more than one ABI is specified, Bazel will build an APK containing a shared
object for each ABI.

Depending on the NDK revision and Android API level, the following ABIs are
available:

| NDK revision | ABIs                                                        |
|--------------|-------------------------------------------------------------|
| 16 and lower | armeabi, armeabi-v7a, arm64-v8a, mips, mips64, x86, x86\_64 |
| 17 and above | armeabi-v7a, arm64-v8a, x86, x86\_64                        |

See [the NDK docs](https://developer.android.com/ndk/guides/abis.html){: .external}
for more information on these ABIs.

Multi-ABI Fat APKs are not recommended for release builds since they increase
the size of the APK, but can be useful for development and QA builds.

## Selecting a C++ standard {:#selecting-c-standard}

Use the following flags to build according to a C++ standard:

| C++ Standard | Flag                    |
|--------------|-------------------------|
| C++98        | Default, no flag needed |
| C++11        | `--cxxopt=-std=c++11`   |
| C++14        | `--cxxopt=-std=c++14`   |

For example:

```posix-terminal
bazel build //:app --cxxopt=-std=c++11
```

Read more about passing compiler and linker flags with `--cxxopt`, `--copt`, and
`--linkopt` in the [User Manual](/docs/user-manual#cxxopt).

Compiler and linker flags can also be specified as attributes in `cc_library`
using `copts` and `linkopts`. For example:

```python
cc_library(
    name = "jni_lib",
    srcs = ["cpp/native-lib.cpp"],
    copts = ["-std=c++11"],
    linkopts = ["-ldl"], # link against libdl
)
```

## Integration with platforms and toolchains {:#integration-platforms}

Bazel's configuration model is moving towards
[platforms](/extending/platforms) and
[toolchains](/extending/toolchains). If your
build uses the `--platforms` flag to select for the architecture or operating system
to build for, you will need to pass the `--extra_toolchains` flag to Bazel in
order to use the NDK.

For example, to integrate with the `android_arm64_cgo` toolchain provided by
the Go rules, pass `--extra_toolchains=@androidndk//:all` in addition to the
`--platforms` flag.

```posix-terminal
bazel build //my/cc:lib \
  --platforms=@io_bazel_rules_go//go/toolchain:android_arm64_cgo \
  --extra_toolchains=@androidndk//:all
```

You can also register them directly in the `WORKSPACE` file:

```python
android_ndk_repository(name = "androidndk")
register_toolchains("@androidndk//:all")
```

Registering these toolchains tells Bazel to look for them in the NDK `BUILD`
file (for NDK 20) when resolving architecture and operating system constraints:

```python
toolchain(
  name = "x86-clang8.0.7-libcpp_toolchain",
  toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
  target_compatible_with = [
      "@platforms//os:android",
      "@platforms//cpu:x86_32",
  ],
  toolchain = "@androidndk//:x86-clang8.0.7-libcpp",
)

toolchain(
  name = "x86_64-clang8.0.7-libcpp_toolchain",
  toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
  target_compatible_with = [
      "@platforms//os:android",
      "@platforms//cpu:x86_64",
  ],
  toolchain = "@androidndk//:x86_64-clang8.0.7-libcpp",
)

toolchain(
  name = "arm-linux-androideabi-clang8.0.7-v7a-libcpp_toolchain",
  toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
  target_compatible_with = [
      "@platforms//os:android",
      "@platforms//cpu:arm",
  ],
  toolchain = "@androidndk//:arm-linux-androideabi-clang8.0.7-v7a-libcpp",
)

toolchain(
  name = "aarch64-linux-android-clang8.0.7-libcpp_toolchain",
  toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
  target_compatible_with = [
      "@platforms//os:android",
      "@platforms//cpu:aarch64",
  ],
  toolchain = "@androidndk//:aarch64-linux-android-clang8.0.7-libcpp",
)
```

## How it works: introducing Android configuration transitions {:#intro-android-config}

The `android_binary` rule can explicitly ask Bazel to build its dependencies in
an Android-compatible configuration so that the Bazel build *just works* without
any special flags, except for `--fat_apk_cpu` and `--android_crosstool_top` for
ABI and STL configuration.

Behind the scenes, this automatic configuration uses Android [configuration
transitions](/extending/rules#configurations).

A compatible rule, like `android_binary`, automatically changes the
configuration of its dependencies to an Android configuration, so only
Android-specific subtrees of the build are affected. Other parts of the build
graph are processed using the top-level target configuration. It may even
process a single target in both configurations, if there are paths through the
build graph to support that.

Once Bazel is in an Android-compatible configuration, either specified at the
top level or due to a higher-level transition point, additional transition
points encountered do not further modify the configuration.

The only built-in location that triggers the transition to the Android
configuration is `android_binary`'s `deps` attribute.

Note: The `data` attribute of `android_binary` intentionally does *not*
trigger the transition. Additionally, `android_local_test` and `android_library`
intentionally do *not* trigger the transition at all.

For example, if you try to build an `android_library` target with a `cc_library`
dependency without any flags, you may encounter an error about a missing JNI
header:

```
ERROR: {{ "<var>" }}project{{ "</var>" }}/app/src/main/BUILD.bazel:16:1: C++ compilation of rule '//app/src/main:jni_lib' failed (Exit 1)
app/src/main/cpp/native-lib.cpp:1:10: fatal error: 'jni.h' file not found
#include <jni.h>
         ^~~~~~~
1 error generated.
Target //app/src/main:lib failed to build
Use --verbose_failures to see the command lines of failed build steps.
```

Ideally, these automatic transitions should make Bazel do the right thing in the
majority of cases. However, if the target on the Bazel command-line is already
below any of these transition rules, such as C++ developers testing a specific
`cc_library`, then a custom `--crosstool_top` must be used.

## Building a `cc_library` for Android without using `android_binary` {:#cclibrary-android}

To build a standalone `cc_binary` or `cc_library` for Android without using an
`android_binary`, use the `--crosstool_top`, `--cpu` and `--host_crosstool_top`
flags.

For example:

```posix-terminal
bazel build //my/cc/jni:target \
      --crosstool_top=@androidndk//:default_crosstool \
      --cpu=<abi> \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
```

In this example, the top-level `cc_library` and `cc_binary` targets are built
using the NDK toolchain. However, this causes Bazel's own host tools to be built
with the NDK toolchain (and thus for Android), because the host toolchain is
copied from the target toolchain. To work around this, specify the value of
`--host_crosstool_top` to be `@bazel_tools//tools/cpp:toolchain` to
explicitly set the host's C++ toolchain.

With this approach, the entire build tree is affected.

Note: All of the targets on the command line must be compatible with
building for Android when specifying these flags, which may make it difficult to
use [Bazel wild-cards](/run/build#specifying-build-targets) like
`/...` and `:all`.

These flags can be put into a `bazelrc` config (one for each ABI), in
`{{ "<var>" }}project{{ "</var>" }}/.bazelrc`:

```
common:android_x86 --crosstool_top=@androidndk//:default_crosstool
common:android_x86 --cpu=x86
common:android_x86 --host_crosstool_top=@bazel_tools//tools/cpp:toolchain

common:android_armeabi-v7a --crosstool_top=@androidndk//:default_crosstool
common:android_armeabi-v7a --cpu=armeabi-v7a
common:android_armeabi-v7a --host_crosstool_top=@bazel_tools//tools/cpp:toolchain

# In general
common:android_<abi> --crosstool_top=@androidndk//:default_crosstool
common:android_<abi> --cpu=<abi>
common:android_<abi> --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
```

Then, to build a `cc_library` for `x86` for example, run:

```posix-terminal
bazel build //my/cc/jni:target --config=android_x86
```

In general, use this method for low-level targets (like `cc_library`) or when
you know exactly what you're building; rely on the automatic configuration
transitions from `android_binary` for high-level targets where you're expecting
to build a lot of targets you don't control.
