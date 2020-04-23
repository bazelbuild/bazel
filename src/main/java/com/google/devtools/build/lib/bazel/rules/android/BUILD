load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]) + [
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "android",
    srcs = glob(["*.java"]),
    resources = glob(
        [
            "*.txt",
            "*.WORKSPACE",
        ],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools:android_ndk_crosstools",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/cpp:bazel_cpp_semantics",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/java",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/java:bazel_java_semantics",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:build_type",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//src/main/java/com/google/devtools/build/lib/rules/android",
        "//src/main/java/com/google/devtools/build/lib/rules/config",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:shell_escaper",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/protobuf:crosstool_config_java_proto",
        "//third_party:android_common_25_0_0_lite",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)
