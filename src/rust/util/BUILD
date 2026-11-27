load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/rust:__pkg__"],
)

rust_library(
    name = "util",
    srcs = ["util.rs"],
)

rust_library(
    name = "util_ffi",
    srcs = ["util_ffi.rs"],
    deps = [":util"],
)

rust_test(
    name = "util_test",
    srcs = ["util_test.rs"],
    deps = [
        ":util",
        "@crates//googletest",
    ],
)
