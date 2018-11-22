package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
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
    visibility = [
        ":bazel-prod",
        ":bazel-tests",
    ],
    exports = [
        ":file",
        ":processes",
    ],
)

java_library(
    name = "file",
    srcs = ["WindowsFileOperations.java"],
    visibility = [":android-junctions-prod"],
    deps = [":jni-loader"],
)

java_library(
    name = "processes",
    srcs = ["WindowsProcesses.java"],
    visibility = [":bazel-prod"],
    deps = [":jni-loader"],
)

java_library(
    name = "jni-loader",
    srcs = ["WindowsJniLoader.java"],
    deps = ["@bazel_tools//tools/java/runfiles"],
)
