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
    name = "tracking_allocator",
    srcs = ["tracking_allocator.rs"],
)

rust_library(
    name = "tracking_allocator_ffi",
    srcs = ["tracking_allocator_ffi.rs"],
    deps = [":tracking_allocator"],
    alwayslink = True,
)

rust_test(
    name = "tracking_allocator_test",
    srcs = ["tracking_allocator_test.rs"],
    # Tests inspect the process-wide global allocation counter.
    args = ["--test-threads=1"],
    deps = [
        ":tracking_allocator",
        ":tracking_allocator_ffi",
        "@crates//googletest",
    ],
)
