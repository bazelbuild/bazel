load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

package_group(
    name = "android-junctions-prod",
    packages = [
        "//src/tools/android/java/com/google/devtools/build/android/junctions",
    ],
)

package_group(
    name = "bazel-prod",
    packages = [
        "//src/main/java/com/google/devtools/build/lib/...",
    ],
)

package_group(
    name = "bazel-tests",
    packages = [
        "//src/test/java/com/google/devtools/build/lib/...",
    ],
)

java_library(
    name = "jni",
    exports = [
        ":file",
        ":processes",
    ],
)

java_library(
    name = "file",
    srcs = ["WindowsFileOperations.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib/jni"],
)

java_library(
    name = "processes",
    srcs = ["WindowsProcesses.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib/jni"],
)
