# Description:
#   Tests for the extension classes for desugaring try-with-resources for Android.
package(
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
)

java_import(
    name = "test_subjects",
    jars = ["test_subjects.jar"],
)
