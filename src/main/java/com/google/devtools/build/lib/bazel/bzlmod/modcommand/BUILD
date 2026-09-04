load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:__pkg__"],
)

java_library(
    name = "modcommand",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:common",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:inspection",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:module_extension",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repo_definition",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repo_definition_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:label_printer",
        "//src/main/java/com/google/devtools/build/lib/util:maybe_complete_set",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:build_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/java/auto:auto_value",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/gson",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/protobuf",
        "//third_party/java/protobuf/util:json",
    ],
)
