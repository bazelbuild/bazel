filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

sh_test(
    name = "android_integration_test",
    size = "large",
    srcs = ["android_integration_test.sh"],
    data = [
        "//src/test/shell/bazel:test-deps",
        # These targets should point to a filegroup with one single item: the
        # root directory of the NDK and the SDK. Note that this is incorrect,
        # but since they contain files with names that are not legal Bazel
        # labels, there isn't really a better option.
        "//external:android_ndk_for_testing",
        "//external:android_sdk_for_testing",
        "//src/tools/android/java/com/google/devtools/build/android:ResourceProcessorBusyBox_deploy.jar",
    ],
)
