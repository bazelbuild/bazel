load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "windows_short_path",
    srcs = ["WindowsShortPath.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib/windows/jni"],  
)

java_library(
    name = "windows",
    srcs = [
        "WindowsFileSystem.java",
        "WindowsSubprocess.java",
        "WindowsSubprocessFactory.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/windows/jni",
        "//third_party:guava",
    ],
)
