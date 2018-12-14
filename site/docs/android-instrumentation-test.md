---
layout: documentation
title: Android Instrumentation Tests
---

# Android Instrumentation Tests

_If you're new to Bazel, please start with the [Building Android with
Bazel](tutorial/android-app.html)
tutorial._

![Running Android instrumentation tests in parallel](/assets/android_test.gif)

[`android_instrumentation_test`](be/android.html#android_instrumentation_test)
allows developers to test their apps on Android emulators and devices.
It utilizes real Android framework APIs and the Android Test Library.

For hermeticity and reproducibility, Bazel creates and launches Android
emulators in a sandbox, ensuring that tests always run from a clean state. Each
test gets an isolated emulator instance, allowing tests to run in parallel
without passing states between them.

For more information on Android instrumentation tests, check out the [Android
developer
documentation](https://developer.android.com/training/testing/unit-testing/instrumented-unit-tests.html).

Please file issues in the [GitHub issue tracker](https://github.com/bazelbuild/bazel/issues).

**Table of Contents**

- [How it works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Getting started](#getting-started)
    - [`BUILD` file](#build-file)
    - [`WORKSPACE` dependencies](#workspace-dependencies)
- [Maven dependencies](#maven-dependencies)
- [Choosing an `android_device` target](#choosing-an-android_device-target)
- [Running tests](#running-tests)
    - [Headless testing](#headless-testing)
    - [GUI testing](#gui-testing)
    - [Testing with a local emulator or device](#testing-with-a-local-emulator-or-device)
- [Sample projects](#sample-projects)
- [Espresso setup](#espresso-setup)
- [Tips](#tips)
    - [Reading test logs](#reading-test-logs)
    - [Testing against multiple API levels](#testing-against-multiple-api-levels)
- [Known issues](#known-issues)
- [Planned features](#planned-features)

# How it works

When you run `bazel test` on an `android_instrumentation_test` target for the
first time, Bazel performs the following steps:

1. Builds the test APK, APK under test, and their transitive dependencies
2. Creates, boots, and caches clean emulator states
3. Starts the emulator
4. Installs the APKs
5. Runs tests utilizing the [Android Test Orchestrator](https://developer.android.com/training/testing/junit-runner.html#using-android-test-orchestrator)
6. Shuts down the emulator
7. Reports the results

In subsequent test runs, Bazel boots the emulator from the clean, cached state
created in step 2, so there are no leftover states from previous runs. Caching
emulator state also speeds up test runs.

# Prerequisites

Ensure your enivornment satisfies the following prerequisites:

- **Linux**. Tested on Ubuntu 14.04 and 16.04.

- **Bazel 0.12.0** or later. Verify the version by running `bazel info release`.

```
$ bazel info release
release 0.12.0
```

- **KVM**. Bazel requires emulators to have [hardware
  acceleration](https://developer.android.com/studio/run/emulator-acceleration.html#accel-check)
  with KVM on Linux. You can follow these
  [installation instructions](https://help.ubuntu.com/community/KVM/Installation)
  for Ubuntu. Run `apt-get install cpu-checker && kvm-ok` to verify that KVM has
  the correct configuration. If it prints the following message, you're good to
  go:

```
$ kvm-ok
INFO: /dev/kvm exists
KVM acceleration can be used
```

- **Xvfb**. To run headless tests (for example, on CI servers), Bazel requires
  the [X virtual framebuffer](https://www.x.org/archive/X11R7.6/doc/man/man1/Xvfb.1.xhtml).
  Install it by running `apt-get install xvfb`. Verify that `Xvfb` is installed
  correctly by running `which Xvfb` and ensure that it's installed at
  `/usr/bin/Xvfb`:

```
$ which Xvfb
/usr/bin/Xvfb
```

# Getting started

Here is a typical target dependency graph of an `android_instrumentation_test`:

![The target dependency graph on an Android instrumentation test](/assets/android_instrumentation_test.png)

## `BUILD` file

The graph translates into a `BUILD` file like this:

```python
load("@gmaven_rules//:defs.bzl", "gmaven_artifact")

android_instrumentation_test(
    name = "my_test",
    test_app = ":my_test_app",
    target_device = "@android_test_support//tools/android/emulated_devices/generic_phone:android_23_x86_qemu2",
)

# Test app and library
android_binary(
    name = "my_test_app",
    instruments = ":my_app",
    manifest = "AndroidTestManifest.xml",
    deps = [":my_test_lib"],
    # ...
)

android_library(
    name = "my_test_lib",
    srcs = glob(["javatest/**/*.java"]),
    deps = [
        ":my_app_lib",
        gmaven_artifact("com.android.support.test.espresso:espresso_core:aar:3.0.1"),
    ],
    # ...
)

# Target app and library under test
android_binary(
    name = "my_app",
    manifest = "AndroidManifest.xml",
    deps = [":my_app_lib"],
    # ...
)

android_library(
    name = "my_app_lib",
    srcs = glob(["java/**/*.java"]),
    deps = [
        gmaven_artifact("com.android.support:design:aar:27.0.2"),
        gmaven_artifact("com.android.support:support_annotations:jar:27.0.2"),
    ]
    # ...
)
```

The main attributes of the rule `android_instrumentation_test` are:

- `test_app`: An `android_binary` target. This target contains test code and
  dependencies like Espresso and UIAutomator. The selected `android_binary`
  target is required to specify an `instruments` attribute pointing to another
  `android_binary`, which is the app under test.

- `target_device`: An `android_device` target. This target describes the
  specifications of the Android emulator which Bazel uses to create, launch and
  run the tests. See the [section on choosing an Android
  device](#choosing-an-android_device) for more information.

The test app's `AndroidManifest.xml` must include
[an `<instrumentation>` tag](https://developer.android.com/studio/test/#configure_instrumentation_manifest_settings).
This tag must specify the attributes for the **package of the target app** and
the **fully qualified class name of the instrumentation test runner**,
`android.support.test.runner.AndroidJUnitRunner`.

Here is an example `AndroidTestManifest.xml` for the test app:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
          xmlns:tools="http://schemas.android.com/tools"
          package="com.example.android.app.test"
    android:versionCode="1"
    android:versionName="1.0">

    <instrumentation
        android:name="android.support.test.runner.AndroidJUnitRunner"
        android:targetPackage="com.example.android.app" />

    <uses-sdk
        android:minSdkVersion="16"
        android:targetSdkVersion="27" />

    <application >
       <!-- ... -->
    </application>
</manifest>
```

## `WORKSPACE` dependencies

In order to use this rule, your project needs to depend on these external
repositories:

- `@androidsdk`: The Android SDK. Download this through Android Studio.

- `@android_test_support`: Hosts the test runner, emulator launcher, and
  `android_device` targets.

- `@gmaven_rules`: Defines the `maven_jar` and `maven_aar` targets available on
  the [Google Maven repository](https://maven.google.com).

You can enable these dependencies by adding the following lines to your
`WORKSPACE` file:

```python
# Android SDK
android_sdk_repository(
    name = "androidsdk",
    path = "/path/to/sdk", # or set ANDROID_HOME
)

# Android Test Support
ATS_COMMIT = "$COMMIT_HASH"
http_archive(
    name = "android_test_support",
    strip_prefix = "android-test-%s" % ATS_COMMIT,
    urls = ["https://github.com/android/android-test/archive/%s.tar.gz" % ATS_COMMIT],
)
load("@android_test_support//:repo.bzl", "android_test_repositories")
android_test_repositories()

# Google Maven Repository
GMAVEN_TAG = "0.1.0"
http_archive(
    name = "gmaven_rules",
    strip_prefix = "gmaven_rules-%s" % GMAVEN_TAG,
    urls = ["https://github.com/bazelbuild/gmaven_rules/archive/%s.tar.gz" % GMAVEN_TAG],
)
load("@gmaven_rules//:gmaven.bzl", "gmaven_rules")
gmaven_rules()
```

# Maven dependencies

Use the
[maven_jar](be/workspace.html#maven_jar)
repository rule for Maven dependencies not hosted on Google Maven. For example,
to use JUnit 4.12 and Hamcrest 2, add the following lines to your `WORKSPACE`:

```
maven_jar(
    name = "junit_junit",
    artifact = "junit:junit:4.12",
)

maven_jar(
    name = "org_hamcrest_java_hamcrest",
    artifact = "org.hamcrest:java-hamcrest:2.0.0.0",
)
```

Then, you can depend on them in your `BUILD` files:

```python
java_library(
    name = "test_deps",
    visibility = ["//visibility:public"],
    exports = [
        "@junit_junit//jar",
        "@org_hamcrest_java_hamcrest//jar",
    ],
)

android_library(
    name = "my_test_lib",
    srcs = [..],
    deps = [":test_deps"],
)
```

[`bazel-deps`](https://github.com/johnynek/bazel-deps) is another useful tool
for managing Maven dependencies using [a `YAML`
file](https://github.com/johnynek/bazel-deps/blob/master/dependencies.yaml).

For dependencies hosted on [Google's Maven
repository](https://maven.google.com), [`@gmaven_rules`](https://github.com/bazelbuild/gmaven_rules)
provides a simple way to fetch dependencies hosted with `gmaven_artifact`.

`gmaven_artifact` is a macro that maps an artifact's coordinate to the actual
generated target in
[`gmaven.bzl`](https://raw.githubusercontent.com/bazelbuild/gmaven_rules/master/gmaven.bzl)
(warning: big file!). The packaging type defaults to `jar` if it isn't
specified.

Load the `gmaven_artifact` macro at the beginning of your `BUILD` file to use
it:

```python
load("@gmaven_rules//:defs.bzl", "gmaven_artifact")

android_library(
    name = "my_app_lib",
    srcs = glob(["java/**/*.java"]),
    deps = [
        gmaven_artifact("com.android.support:design:aar:27.0.2"),
        gmaven_artifact("com.android.support:support_annotations:jar:27.0.2"),
    ]
    # ...
)
```

# Choosing an android_device target

`android_instrumentation_test.target_device` specifies which Android device to
run the tests on. These `android_device` targets are defined in
[`@android_test_support`](https://github.com/google/android-testing-support-library/tree/master/tools/android/emulated_devices).

```python
$ bazel query --output=build @android_test_support//tools/android/emulated_devices/generic_phone:android_23_x86_qemu2
# .../external/android_test_support/tools/android/emulated_devices/generic_phone/BUILD:43:1
android_device(
  name = "android_23_x86_qemu2",
  visibility = ["//visibility:public"],
  tags = ["requires-kvm"],
  generator_name = "generic_phone",
  generator_function = "make_device",
  generator_location = "tools/android/emulated_devices/generic_phone/BUILD:43",
  vertical_resolution = 800,
  horizontal_resolution = 480,
  ram = 2048,
  screen_density = 240,
  cache = 32,
  vm_heap = 256,
  system_image = "@android_test_support//tools/android/emulated_devices/generic_phone:android_23_x86_qemu2_images",
  default_properties = "@android_test_support//tools/android/emulated_devices/generic_phone:_android_23_x86_qemu2_props",
)
```

The device target names use this template:

```
@android_test_support//tools/android/emulated_devices/${device_type}:${system}_${api_level}_x86_qemu2
```

In order to launch an `android_device`, the `system_image` for the selected API
level is required. To download the system image, use Android SDK's
`tools/bin/sdkmanager`. For example, to download the system image for
`generic_phone:android_23_x86_qemu2`, run `$sdk/tools/bin/sdkmanager
"system-images;android-23;default;x86"`.

To see the full list of supported `android_device` targets in
`@android_test_support`, run the following command:

```
bazel query 'filter("x86_qemu2$", kind(android_device, @android_test_support//tools/android/emulated_devices/...:*))'
```

Bazel currently supports x86-based emulators only. For better performance,
we also recommend using `QEMU2` `android_device` targets instead of `QEMU` ones.

# Running tests

To run tests, add these lines to your project's `tools/bazel.rc` file.

```
# Configurations for testing with Bazel
# Select a configuration by running
# `bazel test //my:target --config={headless, gui, local_device}`

# Headless instrumentation tests
test:headless --test_arg=--enable_display=false

# Graphical instrumentation tests. Ensure that $DISPLAY is set.
test:gui --test_env=DISPLAY
test:gui --test_arg=--enable_display=true

# Testing with a local emulator or device. Ensure that `adb devices` lists the
# device.
# Run tests serially.
test:local_device --test_strategy=exclusive
# Use the local device broker type, as opposed to WRAPPED_EMULATOR.
test:local_device --test_arg=--device_broker_type=LOCAL_ADB_SERVER
# Uncomment and set $device_id if there is more than one connected device.
# test:local_device --test_arg=--device_serial_number=$device_id
```

Then, use one of the configurations to run tests:

- `bazel test //my/test:target --config=headless`
- `bazel test //my/test:target --config=gui`
- `bazel test //my/test:target --config=local_device`

Use __only one configuration__ or tests will fail.

## Headless testing

With `Xvfb`, it is possible to test with emulators without the graphical
interface, also known as headless testing. To disable the graphical interface
when running tests, pass the test argument `--enable_display=false` to Bazel:

```
bazel test //my/test:target --test_arg=--enable_display=false
```

## GUI testing

If the `$DISPLAY` environment variable is set, it's possible to enable the
graphical interface of the emulator while the test is running. To do this, pass
these test arguments to Bazel:

```
bazel test //my/test:target --test_arg=--enable_display --test_env=DISPLAY
```

## Testing with a local emulator or device

Bazel also supports testing directly on a locally launched emulator or connected
device. Pass the flags
`--test_strategy=exclusive` and
`--test_arg=--device_broker_type=LOCAL_ADB_SERVER` to enable local testing mode.
If there is more than one connected device, pass the flag
`--test_arg=--device_serial_number=$device_id` where `$device_id` is the id of
the device/emulator listed in `adb devices`.

# Sample projects

If you are looking for canonical project samples, see the [Android testing
samples](https://github.com/googlesamples/android-testing#experimental-bazel-support)
for projects using Espresso and UIAutomator.

```
$ git clone https://github.com/googlesamples/android-testing && cd android-testing
# Set path to Android SDK in WORKSPACE
$ bazel test //ui/... --config=headless
INFO: Analysed 45 targets (1 packages loaded).
INFO: Found 36 targets and 9 test targets...

...

INFO: Elapsed time: 195.665s, Critical Path: 195.22s
INFO: Build completed successfully, 417 total actions
//ui/espresso/BasicSample:BasicSampleInstrumentationTest                 PASSED in 103.7s
//ui/espresso/CustomMatcherSample:CustomMatcherSampleInstrumentationTest PASSED in 113.2s
//ui/espresso/DataAdapterSample:DataAdapterSampleInstrumentationTest     PASSED in 110.2s
//ui/espresso/IdlingResourceSample:IdlingResourceSampleInstrumentationTest PASSED in 102.3s
//ui/espresso/IntentsAdvancedSample:IntentsAdvancedSampleInstrumentationTest PASSED in 98.3s
//ui/espresso/IntentsBasicSample:IntentsBasicSampleInstrumentationTest   PASSED in 103.3s
//ui/espresso/MultiWindowSample:MultiWindowSampleInstrumentationTest     PASSED in 108.3s
//ui/espresso/RecyclerViewSample:RecyclerViewSampleInstrumentationTest   PASSED in 102.9s
//ui/uiautomator/BasicSample:BasicSampleInstrumentationTest              PASSED in 122.6s
```

# Espresso setup

If you write UI tests with [Espresso](https://developer.android.com/training/testing/espresso/)
(`androidx.test.espresso`), you can use the following snippets to set up your
Bazel workspace with the list of commonly used Espresso artifacts and their
dependencies:

```
androidx.test.espresso:espresso-core
androidx.test:rules
androidx.test:runner
javax.inject:javax.inject
org.hamcrest:java-hamcrest
junit:junit
```

One way to organize these dependencies is to create a `//:test_deps` shared
library:

```python
# In <project root>/BUILD.bazel

load("@gmaven_rules//:defs.bzl", "gmaven_artifact")

java_library(
    name = "test_deps",
    visibility = ["//visibility:public"],
    exports = [
        gmaven_artifact("androidx.test.espresso:espresso-core:aar:3.1.0-alpha4"),
        gmaven_artifact("androidx.test:rules:aar:1.1.0-alpha4"),
        gmaven_artifact("androidx.test:runner:aar:1.1.0-alpha4"),
        "@javax_inject_javax_inject//jar",
        "@junit_junit//jar",
        "@org_hamcrest_java_hamcrest//jar",
    ],
)
```

Then, add the required dependencies in `<project root>/WORKSPACE`:


```python
android_sdk_repository(
    name = "androidsdk",
)

# Android Test Support
ATS_COMMIT = "e39a8c7769a5c8b498d0deb0deef3a25b289d410"

http_archive(
    name = "android_test_support",
    strip_prefix = "android-test-%s" % ATS_COMMIT,
    urls = ["https://github.com/android/android-test/archive/%s.tar.gz" % ATS_COMMIT],
)

load("@android_test_support//:repo.bzl", "android_test_repositories")

android_test_repositories()

# Google Maven Repository
GMAVEN_TAG = "20180927-1"

http_archive(
    name = "gmaven_rules",
    strip_prefix = "gmaven_rules-%s" % GMAVEN_TAG,
    url = "https://github.com/bazelbuild/gmaven_rules/archive/%s.tar.gz" % GMAVEN_TAG,
)

load("@gmaven_rules//:gmaven.bzl", "gmaven_rules")

gmaven_rules()

maven_jar(
    name = "junit_junit",
    artifact = "junit:junit:4.12",
)

maven_jar(
    name = "javax_inject_javax_inject",
    artifact = "javax.inject:javax.inject:1",
)

maven_jar(
    name = "org_hamcrest_java_hamcrest",
    artifact = "org.hamcrest:java-hamcrest:2.0.0.0",
)
```

Finally, in your test `android_binary` target, add the `//:test_deps`
dependency:

```python
android_binary(
    name = "my_test_app",
    instruments = "//path/to:app",
    deps = [
        "//:test_deps",
        # ...
    ],
    # ...
)
```

# Tips

## Reading test logs

Use `--test_output=errors` to print logs for failing tests, or
`--test_output=all` to print all test output. If you're looking for an
individual test log, go to
`$PROJECT_ROOT/bazel-testlogs/path/to/InstrumentationTestTargetName`.

For example, the test logs for `BasicSample` canonical project are in
`bazel-testlogs/ui/espresso/BasicSample/BasicSampleInstrumentationTest`:

```
$ tree bazel-testlogs/ui/espresso/BasicSample/BasicSampleInstrumentationTest
.
├── adb.409923.log
├── broker_logs
│   ├── aapt_binary.10.ok.txt
│   ├── aapt_binary.11.ok.txt
│   ├── adb.12.ok.txt
│   ├── adb.13.ok.txt
│   ├── adb.14.ok.txt
│   ├── adb.15.fail.txt
│   ├── adb.16.ok.txt
│   ├── adb.17.fail.txt
│   ├── adb.18.ok.txt
│   ├── adb.19.fail.txt
│   ├── adb.20.ok.txt
│   ├── adb.21.ok.txt
│   ├── adb.22.ok.txt
│   ├── adb.23.ok.txt
│   ├── adb.24.fail.txt
│   ├── adb.25.ok.txt
│   ├── adb.26.fail.txt
│   ├── adb.27.ok.txt
│   ├── adb.28.fail.txt
│   ├── adb.29.ok.txt
│   ├── adb.2.ok.txt
│   ├── adb.30.ok.txt
│   ├── adb.3.ok.txt
│   ├── adb.4.ok.txt
│   ├── adb.5.ok.txt
│   ├── adb.6.ok.txt
│   ├── adb.7.ok.txt
│   ├── adb.8.ok.txt
│   ├── adb.9.ok.txt
│   ├── android_23_x86_qemu2.1.ok.txt
│   └── exec-1
│       ├── adb-2.txt
│       ├── emulator-2.txt
│       └── mksdcard-1.txt
├── device_logcat
│   └── logcat1635880625641751077.txt
├── emulator_itCqtc.log
├── outputs.zip
├── pipe.log.txt
├── telnet_pipe.log.txt
└── tmpuRh4cy
    ├── watchdog.err
    └── watchdog.out

4 directories, 41 files
```

## Reading emulator logs

The emulator logs for `android_device` targets are stored in the `/tmp/`
directory with the name `emulator_xxxxx.log`, where `xxxxx` is a
randomly-generated sequence of characters.

Use this command to find the latest emulator log:

```
ls -1t /tmp/emulator_*.log | head -n 1
```

## Testing against multiple API levels

If you would like to test against multiple API levels, you can use a list
comprehension to create test targets for each API level. For example:

```python
API_LEVELS = [
    "19",
    "20",
    "21",
    "22",
]

[android_instrumentation_test(
    name = "my_test_%s" % API_LEVEL,
    test_app = ":my_test_app",
    target_device = "@android_test_support//tools/android/emulated_devices/generic_phone:android_%s_x86_qemu2" % API_LEVEL,
) for API_LEVEL in API_LEVELS]
```

# Known issues

- [Forked adb server processes are not terminated after
  tests](https://github.com/bazelbuild/bazel/issues/4853)
- While APK building works on all platforms (Linux, macOS, Windows), testing
  only works on Linux.
- Even with `--config=local_adb`, users still need to specify
  `android_instrumentation_test.target_device`.
- If using a local device or emulator, Bazel does not uninstall the APKs after
  the test. Clean the packages by running this command: `adb shell pm list
  packages com.example.android.testing | cut -d ':' -f 2 | tr -d '\r' | xargs
  -L1 -t adb uninstall`

# Planned features

- Code coverage collection
- macOS support
- Windows support
- Improved external dependency management
- Remote test caching and execution

We are planning to rewrite the Android rules in [Starlark](skylark/concepts.html).
The `android_instrumentation_test` rule will be part of the rewrite, however,
its usage will remain unchanged from the end-user perspective.
