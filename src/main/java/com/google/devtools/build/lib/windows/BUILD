package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "windows_short_path",
    srcs = ["WindowsShortPath.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/vfs:__pkg__",
        "//src/test/java/com/google/devtools/build/lib:__pkg__",
    ],
    deps = ["//src/main/java/com/google/devtools/build/lib/windows/jni"],
)

java_library(
    name = "windows",
    srcs = [
        "WindowsFileSystem.java",
        "WindowsSubprocess.java",
        "WindowsSubprocessFactory.java",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/windows/jni",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)
