load("@bazel_pip_dev_deps//:requirements.bzl", "requirement")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

JDK_VERSIONS = [
    "11",
    "17",
    "21",
]

# To avoid accidentally force fetching those dependencies, we put this target in its own package.
filegroup(
    name = "test_repos",
    srcs = [
        requirement("bazel-runfiles"),  # Fetching the hermetic Python toolchain.
        "@bazel_skylib//:WORKSPACE",
        "@com_google_protobuf//:LICENSE",
        "@remote_coverage_tools//:all_lcov_merger_lib",
        "@remote_java_tools//:GenClass",
        "@remote_java_tools_darwin_arm64//:java_tools/src/tools/singlejar/singlejar_local",
        "@remote_java_tools_darwin_x86_64//:java_tools/src/tools/singlejar/singlejar_local",
        "@remote_java_tools_linux//:java_tools/src/tools/singlejar/singlejar_local",
        "@remote_java_tools_windows//:java_tools/src/tools/singlejar/singlejar_local.exe",
        "@rules_cc//:LICENSE",
        "@rules_java//:WORKSPACE",
        "@rules_license//:WORKSPACE",
        "@rules_pkg//:WORKSPACE",
        "@rules_python//:WORKSPACE",
        "@rules_shell//shell:rules_bzl",
        "@rules_testing//:LICENSE",
    ] + select({
        "//src/conditions:linux_x86_64": ["@remotejdk%s_linux//:BUILD.bazel" % v for v in JDK_VERSIONS],
        "//src/conditions:darwin_x86_64": ["@remotejdk%s_macos//:BUILD.bazel" % v for v in JDK_VERSIONS],
        "//src/conditions:darwin_arm64": ["@remotejdk%s_macos_aarch64//:BUILD.bazel" % v for v in JDK_VERSIONS],
        "//src/conditions:windows_x86_64": ["@remotejdk%s_win//:BUILD.bazel" % v for v in JDK_VERSIONS],
        "//conditions:default": [],
    }) + [
        "@bazel_tools_repo_cache//:files",
    ],
    visibility = ["//src:__subpackages__"],
)
