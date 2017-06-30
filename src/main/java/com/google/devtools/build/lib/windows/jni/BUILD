package(
    default_visibility = [
        "//src/main/java/com/google/devtools/build/lib:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib:__subpackages__",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
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
    deps = [":jni-loader"],
)

java_library(
    name = "processes",
    srcs = ["WindowsProcesses.java"],
    deps = [":jni-loader"],
)

java_library(
    name = "jni-loader",
    srcs = ["WindowsJniLoader.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib/windows/runfiles"],
)
