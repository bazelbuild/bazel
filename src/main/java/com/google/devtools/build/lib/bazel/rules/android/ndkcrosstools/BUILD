load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]) + [
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r10e:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r11:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r12:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r13:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r15:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r17:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r18:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r19:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "android_ndk_crosstools",
    srcs = [
        "AndroidNdkCrosstools.java",
        "NdkRelease.java",
    ],
    deps = [
        ":ndkcrosstools",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r10e",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r11",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r12",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r13",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r15",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r17",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r18",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools/r19",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party:guava",
    ],
)

java_library(
    name = "ndkcrosstools",
    srcs = glob(
        ["*.java"],
        exclude = [
            "AndroidNdkCrosstools.java",
            "NdkRelease.java",
        ],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/protobuf:crosstool_config_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
