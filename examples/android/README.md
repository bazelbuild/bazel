In order to build these examples, add the following two rules to the top-level `WORKSPACE` file (two directories above this file):

```python
android_sdk_repository(
    name="androidsdk",
    path="<full path to your Android SDK>",
    api_level=<api level>,
)

android_ndk_repository(
    name="androidndk",
    path="<path to your Android NDK>",
    api_level=<api_level>,
)
```

For the `android_sdk_repository` rule, the value of `api_level` corresponds to
a directory in the SDK containing the specific version of `android.jar` to
compile against. For example, if `path = "/Users/xyzzy/Library/Android/sdk"` and
`api_level = 21`, then the directory
`/Users/xyzzy/Library/Android/sdk/platforms/android-21` must exist.

Similarly, for the `android_ndk_repository` rule, the value of the `api_level`
attribute corresponds to a directory containing the NDK libraries for that
API level. For example, if
`path=/Users/xyzzy/Library/Android/android-ndk-r10e` and
`api_level=21`, then you your NDK must contain the directory
`/Users/xyzzy/Library/Android/android-ndk-r10e/platforms/android-21`.

The example `android_binary` depends on
`@androidsdk//com.android.support:appcompat-v7-25.0.0`, so you will need to
install the Google Support Libraries version 25.0.0 from the Android SDK
Manager.

The following command can be used to build the example app:

```
bazel build //examples/android/java/bazel:hello_world
```

We also have a nice way to speed up the edit-compile-install development cycle for physical Android devices and emulators: Bazel knows what code changed since the last build, and can use this knowledge to install only the changed code to the device. This currently works with L devices and changes to Java code and Android resources. To try this out, take an `android_binary` rule and:

 * Set the `proguard_specs` attribute to `[]` (the empty list) or just omit it altogether
 * Set the `multidex` attribute to `native`
 * Set the `dex_shards` attribute to a number between 2 and 200. This controls the size of chunks the code is split into. As this number is increased, compilation and installation becomes faster but app startup becomes slower. A good initial guess is 10.
 * Connect your device over USB to your workstation and enable USB debugging on it
 * Run `bazel mobile-install <android_binary rule>`
 * Edit Java code or Android resources
 * Run `bazel mobile-install --incremental <android_binary rule>`

Note that if you change anything other than Java code or Android resources (C++ code or something on the device), you must omit the `--incremental` command line option. Yes, we know that this is also clunky and we are working on improving it.
