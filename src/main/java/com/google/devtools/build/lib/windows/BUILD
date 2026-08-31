load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "file",
    srcs = ["WindowsFileOperations.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
        "//third_party/bazel_rules/rules_android/src/tools/java/com/google/devtools/build/android/junctions:__pkg__",
    ],
    deps = [
        ":windows_path_operations",
        "//src/main/java/com/google/devtools/build/lib/jni",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
    ],
)

java_library(
    name = "processes",
    srcs = ["WindowsProcesses.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib/jni"],
)

java_library(
    name = "windows_path_operations",
    srcs = ["WindowsPathOperations.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib/jni"],
)

java_library(
    name = "windows",
    srcs = [
        "WindowsFileSystem.java",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
        "//src/test/java/com/google/devtools/build/lib/windows:__subpackages__",
    ],
    deps = [
        ":file",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/java/guava:annotations",
        "//third_party/java/jsr305_annotations",
    ],
)
