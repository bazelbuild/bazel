load("@rules_cc//cc:defs.bzl", "cc_test")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

cc_test(
    name = "process-tools_test",
    srcs = ["process-tools_test.cc"] + select({
        "//src/conditions:darwin": ["process-tools-darwin_test.cc"],
        "//conditions:default": ["process-tools-linux_test.cc"],
    }),
    deps = [
        "//src/main/tools:process-tools",
        "@com_google_googletest//:gtest_main",
    ],
)
