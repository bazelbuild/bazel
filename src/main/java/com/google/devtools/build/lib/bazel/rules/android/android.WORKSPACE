bind(
    name = "android/sdk",
    actual = "@bazel_tools//tools/android:poison_pill_android_sdk",
)

bind(
    name = "android/dx_jar_import",
    actual = "@bazel_tools//tools/android:no_android_sdk_repository_error",
)

bind(
    name = "android/crosstool",
    actual = "@bazel_tools//tools/cpp:toolchain",
)

bind(
    name = "android_sdk_for_testing",
    actual = "//:dummy",
)

bind(
    name = "android_ndk_for_testing",
    actual = "//:dummy",
)

bind(
    name = "databinding_annotation_processor",
    actual = "@bazel_tools//tools/android:empty",
)

# This value is overridden by android_sdk_repository function to allow targets
# to select on whether or not android_sdk_repository has run.
bind(
    name = "has_androidsdk",
    actual = "@bazel_tools//tools/android:always_false",
)
