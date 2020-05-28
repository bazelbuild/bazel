# Description:
#   Collection of dex utilities used in the bazel android actions.

load("@rules_java//java:defs.bzl", "java_library")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src/test/java/com/google/devtools/build/android/r8:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(
        ["*.java"],
        exclude = ["NoAndroidSdkStub.java"],
    ) + ["BUILD.tools"],
    visibility = ["//visibility:public"],
)

java_library(
    name = "r8",
    srcs = select({
        "//external:has_androidsdk": glob(
            ["*.java"],
            exclude = [
                "NoAndroidSdkStub.java",
                "Desugar.java",
                "desugar/*.java",
            ],
        ),
        "//conditions:default": ["NoAndroidSdkStub.java"],
    }),
    visibility = ["//src/test/java/com/google/devtools/build/android/r8:__pkg__"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/tools/android/java/com/android/tools/r8:r8_support",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:asm",
        "//third_party:guava",
    ] + select({
        "//external:has_androidsdk": ["//external:android/d8_jar_import"],
        "//conditions:default": [],
    }),
)

java_library(
    name = "utils",
    srcs = ["R8Utils.java"],
    visibility = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar:__subpackages__",
    ],
)

# The DexFileMerger and DexBuilder are built in BUILD.tools which is built in
# a developers workspace, not the Bazel workspace. So we must bundle the
# dependencies of those binaries into the embedded tools.
java_library(
    name = "dexerdeps",
    visibility = [
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
    ],
    runtime_deps = [
        # None right now. The D8/R8 jar has no dependencies.
    ],
)
