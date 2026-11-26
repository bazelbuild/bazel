load("@rules_cc//cc:cc_library.bzl", "cc_library")
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

cc_library(
    name = "darwin_frameworks",
    linkopts = select({
        "@platforms//os:macos": [
            "-framework CoreFoundation",
            "-framework IOKit",
        ],
        "//conditions:default": [],
    }),
)

rust_library(
    name = "sleep_prevention",
    srcs = [
        "sleep_prevention.rs",
        "sleep_prevention_darwin.rs",
        "sleep_prevention_unsupported.rs",
        "sleep_prevention_windows.rs",
    ],
    crate_root = "sleep_prevention.rs",
    deps = [":darwin_frameworks"],
)

rust_library(
    name = "sleep_prevention_ffi",
    srcs = ["sleep_prevention_ffi.rs"],
    deps = [":sleep_prevention"],
    alwayslink = True,
)

rust_test(
    name = "sleep_prevention_test",
    srcs = ["sleep_prevention_test.rs"],
    deps = [
        ":sleep_prevention",
        "@crates//googletest",
    ],
)
