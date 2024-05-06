load("@rules_pkg//pkg:zip.bzl", "pkg_zip")
load("//src:release_archive.bzl", "release_archive")

# Description:
#   one version C++ implementation.
package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

SOURCES = [
    # keep sorted
    "allowlist.cc",
    "allowlist.h",
    "duplicate_class_collector.cc",
    "duplicate_class_collector.h",
    "one_version.cc",
    "one_version.h",
    "one_version_main.cc",
]

cc_library(
    name = "duplicate_class_collector",
    srcs = [
        "duplicate_class_collector.cc",
    ],
    hdrs = [
        "duplicate_class_collector.h",
    ],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/strings",
    ],
)

cc_test(
    name = "duplicate_class_collector_test",
    srcs = [
        "duplicate_class_collector_test.cc",
    ],
    deps = [
        ":duplicate_class_collector",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "allowlist",
    srcs = ["allowlist.cc"],
    hdrs = ["allowlist.h"],
    deps = [
        ":duplicate_class_collector",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "allowlist_test",
    srcs = ["allowlist_test.cc"],
    deps = [
        ":allowlist",
        ":duplicate_class_collector",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "one_version",
    srcs = ["one_version.cc"],
    hdrs = ["one_version.h"],
    deps = [
        ":allowlist",
        ":duplicate_class_collector",
        "//src/tools/singlejar:input_jar",
        "@abseil-cpp//absl/log:die_if_null",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/strings",
    ],
)

cc_binary(
    name = "one_version_main",
    srcs = [
        "one_version_main.cc",
    ],
    deps = [
        ":allowlist",
        ":duplicate_class_collector",
        ":one_version",
        "//src/tools/singlejar:input_jar",
        "//src/tools/singlejar:token_stream",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:die_if_null",
        "@abseil-cpp//absl/strings",
    ],
)

filegroup(
    name = "embedded_java_tools",
    srcs = SOURCES + [
        "//src/tools/singlejar:embedded_java_tools",
    ],
    visibility = ["//src:__pkg__"],
)

release_archive(
    name = "one_version_transitive_zip",
    package_dir = "java_tools",
    visibility = ["//src:__pkg__"],
    deps = [
        ":one_version_zip",
        "//src/tools/singlejar:singlejar_transitive_zip",
    ],
)

release_archive(
    name = "one_version_zip",
    srcs = SOURCES,
    package_dir = "src/tools/one_version",
    visibility = ["//visibility:private"],
)

release_archive(
    name = "one_version_deploy_zip",
    srcs = [":one_version_main"],
    package_dir = "java_tools/src/tools/one_version",
    visibility = ["//src:__pkg__"],
)
