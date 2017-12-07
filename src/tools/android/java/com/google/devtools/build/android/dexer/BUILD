# Description:
#   Collection of dex utilities used in the bazel android actions.

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src/test/java/com/google/devtools/build/android/dexer:__pkg__",
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
    name = "dexer",
    srcs = select({
        "//external:has_androidsdk": glob(
            ["*.java"],
            exclude = ["NoAndroidSdkStub.java"],
        ),
        "//conditions:default": ["NoAndroidSdkStub.java"],
    }),
    visibility = ["//src/test/java/com/google/devtools/build/android/dexer:__pkg__"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ] + select({
        "//external:has_androidsdk": ["//external:android/dx_jar_import"],
        "//conditions:default": [],
    }),
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
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
