Project: /_project.yaml
Book: /_book.yaml

# Android Instrumentation Tests

{% include "_buttons.html" %}

_If you're new to Bazel, start with the [Building Android with
Bazel](/start/android-app ) tutorial._

![Running Android instrumentation tests in parallel](/docs/images/android_test.gif "Android instrumentation test")

**Figure 1.** Running parallel Android instrumentation tests.

[`android_instrumentation_test`](/reference/be/android#android_instrumentation_test)
allows developers to test their apps on Android emulators and devices.
It utilizes real Android framework APIs and the Android Test Library.

For hermeticity and reproducibility, Bazel creates and launches Android
emulators in a sandbox, ensuring that tests always run from a clean state. Each
test gets an isolated emulator instance, allowing tests to run in parallel
without passing states between them.

For more information on Android instrumentation tests, check out the [Android
developer
documentation](https://developer.android.com/training/testing/unit-testing/instrumented-unit-tests.html){: .external}.

Please file issues in the [GitHub issue tracker](https://github.com/bazelbuild/bazel/issues){: .external}.

## How it works {:#how-it-works}

When you run `bazel test` on an `android_instrumentation_test` target for the
first time, Bazel performs the following steps:

1. Builds the test APK, APK under test, and their transitive dependencies
2. Creates, boots, and caches clean emulator states
3. Starts the emulator
4. Installs the APKs
5. Runs tests utilizing the [Android Test Orchestrator](https://developer.android.com/training/testing/junit-runner.html#using-android-test-orchestrator){: .external}
6. Shuts down the emulator
7. Reports the results

In subsequent test runs, Bazel boots the emulator from the clean, cached state
created in step 2, so there are no leftover states from previous runs. Caching
emulator state also speeds up test runs.

## Prerequisites {:#prerequisites}

Ensure your environment satisfies the following prerequisites:

- **Linux**. Tested on Ubuntu 16.04, and 18.04.

- **Bazel 0.12.0** or later. Verify the version by running `bazel info release`.

```posix-terminal
bazel info release
```
This results in output similar to the following:

```none {:.devsite-disable-click-to-copy}
release 4.1.0
```

- **KVM**. Bazel requires emulators to have [hardware
  acceleration](https://developer.android.com/studio/run/emulator-acceleration.html#accel-check){: .external}
  with KVM on Linux. You can follow these
  [installation instructions](https://help.ubuntu.com/community/KVM/Installation){: .external}
  for Ubuntu.

To verify that KVM has the correct configuration, run:

```posix-terminal
apt-get install cpu-checker && kvm-ok
```

If it prints the following message, you have the correct configuration:

```none {:.devsite-disable-click-to-copy}
INFO: /dev/kvm exists
KVM acceleration can be used
```

- **Xvfb**. To run headless tests (for example, on CI servers), Bazel requires
  the [X virtual framebuffer](https://www.x.org/archive/X11R7.6/doc/man/man1/Xvfb.1.xhtml).

To install it, run:

```posix-terminal
apt-get install xvfb
```
Verify that `Xvfb` is installed correctly and is installed at `/usr/bin/Xvfb`
by running:

```posix-terminal
which Xvfb
```
The output is the following:

```{:.devsite-disable-click-to-copy}
/usr/bin/Xvfb
```

- **32-bit Libraries**. Some of the binaries used by the test infrastructure are
  32-bit, so on 64-bit machines, ensure that 32-bit binaries can be run. For
  Ubuntu, install these 32-bit libraries:

```posix-terminal
sudo apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386 lib32z1 libbz2-1.0:i386
```

## Getting started {:#getting-started}

Here is a typical target dependency graph of an `android_instrumentation_test`:

![The target dependency graph on an Android instrumentation test](/docs/images/android_instrumentation_test.png "Target dependency graph")

**Figure 2.** Target dependency graph of an `android_instrumentation_test`.


### BUILD file {:#build-file}

The graph translates into a `BUILD` file like this:

```python
android_instrumentation_test(
    name = "my_test",
    test_app = ":my_test_app",
    target_device = "@android_test_support//tools/android/emulated_devices/generic_phone:android_23_x86",
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
        "@maven//:androidx_test_core",
        "@maven//:androidx_test_runner",
        "@maven//:androidx_test_espresso_espresso_core",
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
        "@maven//:androidx_appcompat_appcompat",
        "@maven//:androidx_annotation_annotation",
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
  device](#android-device-target) for more information.

The test app's `AndroidManifest.xml` must include [an `<instrumentation>`
tag](https://developer.android.com/studio/test/#configure_instrumentation_manifest_settings){: .external}.
This tag must specify the attributes for the **package of the target app** and
the **fully qualified class name of the instrumentation test runner**,
`androidx.test.runner.AndroidJUnitRunner`.

Here is an example `AndroidTestManifest.xml` for the test app:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
          xmlns:tools="http://schemas.android.com/tools"
          package="com.example.android.app.test"
    android:versionCode="1"
    android:versionName="1.0">

    <instrumentation
        android:name="androidx.test.runner.AndroidJUnitRunner"
        android:targetPackage="com.example.android.app" />

    <uses-sdk
        android:minSdkVersion="16"
        android:targetSdkVersion="27" />

    <application >
       <!-- ... -->
    </application>
</manifest>
```

### WORKSPACE dependencies {:#workspace-dependencies}

In order to use this rule, your project needs to depend on these external
repositories:

- `@androidsdk`: The Android SDK. Download this through Android Studio.

- `@android_test_support`: Hosts the test runner, emulator launcher, and
  `android_device` targets. You can find the [latest release
  here](https://github.com/android/android-test/releases){: .external}.

Enable these dependencies by adding the following lines to your `WORKSPACE`
file:

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
```

## Maven dependencies {:#maven-dependencies}

For managing dependencies on Maven artifacts from repositories, such as [Google
Maven](https://maven.google.com){: .external} or [Maven Central](https://central.maven.org){: .external},
you should use a Maven resolver, such as
[`rules_jvm_external`](https://github.com/bazelbuild/rules_jvm_external){: .external}.

The rest of this page shows how to use `rules_jvm_external` to
resolve and fetch dependencies from Maven repositories.

## Choosing an android_device target {:#android-device-target}

`android_instrumentation_test.target_device` specifies which Android device to
run the tests on. These `android_device` targets are defined in
[`@android_test_support`](https://github.com/google/android-testing-support-library/tree/master/tools/android/emulated_devices){: .external}.

For example, you can query for the sources for a particular target by running:

```posix-terminal
bazel query --output=build @android_test_support//tools/android/emulated_devices/generic_phone:android_23_x86
```
Which results in output that looks similar to:

```python
# .../external/android_test_support/tools/android/emulated_devices/generic_phone/BUILD:43:1
android_device(
  name = "android_23_x86",
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
  system_image = "@android_test_support//tools/android/emulated_devices/generic_phone:android_23_x86_images",
  default_properties = "@android_test_support//tools/android/emulated_devices/generic_phone:_android_23_x86_props",
)
```

The device target names use this template:

```
@android_test_support//tools/android/emulated_devices/{{ "<var>" }}device_type{{ "</var>" }}:{{ "<var>" }}system{{ "</var>" }}_{{ "<var>" }}api_level{{ "</var>" }}_x86_qemu2
```

In order to launch an `android_device`, the `system_image` for the selected API
level is required. To download the system image, use Android SDK's
`tools/bin/sdkmanager`. For example, to download the system image for
`generic_phone:android_23_x86`, run `$sdk/tools/bin/sdkmanager
"system-images;android-23;default;x86"`.

To see the full list of supported `android_device` targets in
`@android_test_support`, run the following command:

```posix-terminal
bazel query 'filter("x86_qemu2$", kind(android_device, @android_test_support//tools/android/emulated_devices/...:*))'
```

Bazel currently supports x86-based emulators only. For better performance, use
`QEMU2` `android_device` targets instead of `QEMU` ones.

## Running tests {:#running-tests}

To run tests, add these lines to your project's
`{{ '<var>' }}project root{{ '</var>' }}:{{ '<var>' }}/.bazelrc` file.

```
# Configurations for testing with Bazel
# Select a configuration by running
# `bazel test //my:target --config={headless, gui, local_device}`

# Headless instrumentation tests (No GUI)
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

- `bazel test //my/test:target --config=gui`
- `bazel test //my/test:target --config=headless`
- `bazel test //my/test:target --config=local_device`

Use __only one configuration__ or tests will fail.

### Headless testing {:#headless-testing}

With `Xvfb`, it is possible to test with emulators without the graphical
interface, also known as headless testing. To disable the graphical interface
when running tests, pass the test argument `--enable_display=false` to Bazel:

```posix-terminal
bazel test //my/test:target --test_arg=--enable_display=false
```

### GUI testing {:#gui-testing}

If the `$DISPLAY` environment variable is set, it's possible to enable the
graphical interface of the emulator while the test is running. To do this, pass
these test arguments to Bazel:

```posix-terminal
bazel test //my/test:target --test_arg=--enable_display=true --test_env=DISPLAY
```

### Testing with a local emulator or device {:#testing-local-emulator}

Bazel also supports testing directly on a locally launched emulator or connected
device. Pass the flags
`--test_strategy=exclusive` and
`--test_arg=--device_broker_type=LOCAL_ADB_SERVER` to enable local testing mode.
If there is more than one connected device, pass the flag
`--test_arg=--device_serial_number=$device_id` where `$device_id` is the id of
the device/emulator listed in `adb devices`.

## Sample projects {:#sample-projects}

If you are looking for canonical project samples, see the [Android testing
samples](https://github.com/googlesamples/android-testing#experimental-bazel-support){: .external}
for projects using Espresso and UIAutomator.

## Espresso setup {:#espresso-setup}

If you write UI tests with [Espresso](https://developer.android.com/training/testing/espresso/){: .external}
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
library in your `{{ "<var>" }}project root{{ "</var>" }}/BUILD.bazel` file:

```python
java_library(
    name = "test_deps",
    visibility = ["//visibility:public"],
    exports = [
        "@maven//:androidx_test_espresso_espresso_core",
        "@maven//:androidx_test_rules",
        "@maven//:androidx_test_runner",
        "@maven//:javax_inject_javax_inject"
        "@maven//:org_hamcrest_java_hamcrest",
        "@maven//:junit_junit",
    ],
)
```

Then, add the required dependencies in `{{ "<var>" }}project root{{ "</var>" }}/WORKSPACE`:

```python
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_JVM_EXTERNAL_TAG = "2.8"
RULES_JVM_EXTERNAL_SHA = "79c9850690d7614ecdb72d68394f994fef7534b292c4867ce5e7dec0aa7bdfad"

http_archive(
    name = "rules_jvm_external",
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
    sha256 = RULES_JVM_EXTERNAL_SHA,
    url = "https://github.com/bazelbuild/rules_jvm_external/archive/%s.zip" % RULES_JVM_EXTERNAL_TAG,
)

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    artifacts = [
        "junit:junit:4.12",
        "javax.inject:javax.inject:1",
        "org.hamcrest:java-hamcrest:2.0.0.0"
        "androidx.test.espresso:espresso-core:3.1.1",
        "androidx.test:rules:aar:1.1.1",
        "androidx.test:runner:aar:1.1.1",
    ],
    repositories = [
        "https://maven.google.com",
        "https://repo1.maven.org/maven2",
    ],
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

## Tips {:#tips}

### Reading test logs {:#reading-test-logs}

Use `--test_output=errors` to print logs for failing tests, or
`--test_output=all` to print all test output. If you're looking for an
individual test log, go to
`$PROJECT_ROOT/bazel-testlogs/path/to/InstrumentationTestTargetName`.

For example, the test logs for `BasicSample` canonical project are in
`bazel-testlogs/ui/espresso/BasicSample/BasicSampleInstrumentationTest`, run:

```posix-terminal
tree bazel-testlogs/ui/espresso/BasicSample/BasicSampleInstrumentationTest
```
This results in the following output:

```none

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
│   ├── android_23_x86.1.ok.txt
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

### Reading emulator logs {:#reading-emulator-logs}

The emulator logs for `android_device` targets are stored in the `/tmp/`
directory with the name `emulator_xxxxx.log`, where `xxxxx` is a
randomly-generated sequence of characters.

Use this command to find the latest emulator log:

```posix-terminal
ls -1t /tmp/emulator_*.log | head -n 1
```

### Testing against multiple API levels {:#testing-multiple-apis}

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

## Known issues {:#known-issues}

- [Forked adb server processes are not terminated after
  tests](https://github.com/bazelbuild/bazel/issues/4853){: .external}
- While APK building works on all platforms (Linux, macOS, Windows), testing
  only works on Linux.
- Even with `--config=local_adb`, users still need to specify
  `android_instrumentation_test.target_device`.
- If using a local device or emulator, Bazel does not uninstall the APKs after
  the test. Clean the packages by running this command:

```posix-terminal
adb shell pm list
packages com.example.android.testing | cut -d ':' -f 2 | tr -d '\r' | xargs
-L1 -t adb uninstall
```
