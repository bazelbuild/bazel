package(default_visibility = ["//visibility:private"])

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

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [":bazel-prod"],
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
    deps = ["//src/main/java/com/google/devtools/build/lib/windows/runfiles"],
)
