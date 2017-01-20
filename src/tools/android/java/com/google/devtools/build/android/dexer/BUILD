# Description:
#   Collection of dex utilities used in the bazel android actions.

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/android/java/com/google/devtools/build/android:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["*.java"]) + [
        "BUILD.tools",
    ],
    visibility = ["//visibility:public"],
)
