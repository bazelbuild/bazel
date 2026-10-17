load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/main/java/com/google/devtools/build/lib:build_lib"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "hash",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
        "//src/test/java/com/google/devtools/build/lib/query2/cquery:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/query2/query/hash:__pkg__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:toolchain_type_requirement",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:common",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:exception",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:resolution",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repo_definition",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repo_definition_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/io:inconsistent_filesystem_exception",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:declared_exec_group",
        "//src/main/java/com/google/devtools/build/lib/packages:label_printer",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/query2/query/output",
        "//src/main/java/com/google/devtools/build/lib/rules:repository/repository_directory_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe:package_lookup_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:package_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:repository_mapping_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:target_loading_util",
        "//src/main/java/com/google/devtools/build/lib/skyframe:target_pattern_util",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:build_java_proto",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:hash",
        "//third_party/java/guava:io",
        "//third_party/java/jsr305_annotations",
    ],
)
