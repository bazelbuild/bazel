load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "mobileinstall",
    srcs = glob(["*.java"]),
    resources = glob([
        "*.txt",
    ]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:runtime/blaze_command_result",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/test_configuration",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:google-auto-profiler-utils",
        "//src/main/java/com/google/devtools/build/lib/rules/android",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/util:command",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:command_server_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
