# Description:
#   Tool for desugaring Java constructs not supported by Android tools or devices.

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "desugar_bin_deploy.jar",
    ],
    visibility = ["//src/tools/android/java/com/google/devtools/build/android:__pkg__"],
)

java_library(
    name = "desugar",
    srcs = glob(["*.java"]),
    runtime_deps = ["//src/tools/android/java/com/google/devtools/build/android/desugar/runtime:throwable_extension"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_binary(
    name = "desugar_bin",
    main_class = "does.not.exist",
    runtime_deps = [":desugar"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/runtime:srcs",
    ],
    visibility = ["//src/tools/android/java/com/google/devtools/build/android:__pkg__"],
)
