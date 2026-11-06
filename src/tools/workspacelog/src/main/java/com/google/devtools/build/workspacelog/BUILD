load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/tools/workspacelog:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/workspacelog:__pkg__"],
)

java_library(
    name = "parser",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel/debug:workspace_log_java_proto",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party/java/guava:annotations",
    ],
)
