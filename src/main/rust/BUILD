load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@rules_rs//rs:rust_binary.bzl", "rust_binary")
load("@rules_rs//rs:rust_library.bzl", "rust_library")
load("@rules_rs//rs:rust_shared_library.bzl", "rust_shared_library")
load("@rules_rs//rs:rust_test.bzl", "rust_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

rust_library(
    name = "bazel_proto",
    srcs = glob(["bazel-proto/src/**/*.rs"]),
    crate_name = "bazel_proto",
    edition = "2021",
)

rust_test(
    name = "bazel_proto_test",
    crate = ":bazel_proto",
)

rust_library(
    name = "bazel_util",
    srcs = glob(["bazel-util/src/**/*.rs"]),
    crate_name = "bazel_util",
    edition = "2021",
    deps = [":bazel_proto"],
)

rust_test(
    name = "bazel_util_test",
    crate = ":bazel_util",
)

rust_library(
    name = "bazel_vfs",
    srcs = glob(["bazel-vfs/src/**/*.rs"]),
    crate_name = "bazel_vfs",
    edition = "2021",
    deps = [":bazel_util"],
)

rust_test(
    name = "bazel_vfs_test",
    crate = ":bazel_vfs",
)

rust_library(
    name = "bazel_events",
    srcs = glob(["bazel-events/src/**/*.rs"]),
    crate_name = "bazel_events",
    edition = "2021",
    deps = [
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_events_test",
    crate = ":bazel_events",
)

rust_library(
    name = "bazel_cmdline",
    srcs = glob(["bazel-cmdline/src/**/*.rs"]),
    crate_name = "bazel_cmdline",
    edition = "2021",
    deps = [
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_cmdline_test",
    crate = ":bazel_cmdline",
)

rust_library(
    name = "bazel_options",
    srcs = glob(["bazel-options/src/**/*.rs"]),
    crate_name = "bazel_options",
    edition = "2021",
    deps = [
        ":bazel_proto",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_options_test",
    crate = ":bazel_options",
)

rust_library(
    name = "bazel_starlark",
    srcs = glob(["bazel-starlark/src/**/*.rs"]),
    crate_name = "bazel_starlark",
    edition = "2021",
    deps = [
        ":bazel_cmdline",
        ":bazel_events",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_starlark_test",
    crate = ":bazel_starlark",
)

rust_library(
    name = "bazel_packages",
    srcs = glob(["bazel-packages/src/**/*.rs"]),
    crate_name = "bazel_packages",
    edition = "2021",
    deps = [
        ":bazel_cmdline",
        ":bazel_events",
        ":bazel_starlark",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_packages_test",
    crate = ":bazel_packages",
)

rust_library(
    name = "bazel_bzlmod",
    srcs = glob(["bazel-bzlmod/src/**/*.rs"]),
    crate_name = "bazel_bzlmod",
    edition = "2021",
    deps = [
        ":bazel_cmdline",
        ":bazel_starlark",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_bzlmod_test",
    crate = ":bazel_bzlmod",
)

rust_library(
    name = "bazel_query",
    srcs = glob(["bazel-query/src/**/*.rs"]),
    crate_name = "bazel_query",
    edition = "2021",
    deps = [
        ":bazel_cmdline",
        ":bazel_packages",
        ":bazel_starlark",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_query_test",
    crate = ":bazel_query",
)

rust_library(
    name = "bazel_analysis",
    srcs = glob(["bazel-analysis/src/**/*.rs"]),
    crate_name = "bazel_analysis",
    edition = "2021",
    deps = [
        ":bazel_cmdline",
        ":bazel_packages",
        ":bazel_starlark",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_analysis_test",
    crate = ":bazel_analysis",
)

rust_library(
    name = "bazel_rules",
    srcs = glob(["bazel-rules/src/**/*.rs"]),
    crate_name = "bazel_rules",
    edition = "2021",
    deps = [
        ":bazel_analysis",
        ":bazel_cmdline",
        ":bazel_packages",
        ":bazel_starlark",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_rules_test",
    crate = ":bazel_rules",
)

rust_library(
    name = "bazel_sandbox",
    srcs = glob(["bazel-sandbox/src/**/*.rs"]),
    crate_name = "bazel_sandbox",
    edition = "2021",
    deps = [
        ":bazel_proto",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_sandbox_test",
    crate = ":bazel_sandbox",
)

rust_library(
    name = "bazel_remote",
    srcs = glob(["bazel-remote/src/**/*.rs"]),
    crate_name = "bazel_remote",
    edition = "2021",
    deps = [
        ":bazel_proto",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_remote_test",
    crate = ":bazel_remote",
)

rust_library(
    name = "bazel_exec",
    srcs = glob(["bazel-exec/src/**/*.rs"]),
    crate_name = "bazel_exec",
    edition = "2021",
    deps = [
        ":bazel_analysis",
        ":bazel_cmdline",
        ":bazel_events",
        ":bazel_remote",
        ":bazel_sandbox",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_exec_test",
    crate = ":bazel_exec",
)

rust_library(
    name = "bazel_skyframe",
    srcs = glob(["bazel-skyframe/src/**/*.rs"]),
    crate_name = "bazel_skyframe",
    edition = "2021",
    deps = [
        ":bazel_analysis",
        ":bazel_cmdline",
        ":bazel_packages",
        ":bazel_rules",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_skyframe_test",
    crate = ":bazel_skyframe",
)

rust_library(
    name = "bazel_server",
    srcs = glob(
        ["bazel-server/src/**/*.rs"],
        exclude = ["bazel-server/src/main.rs"],
    ),
    crate_name = "bazel_server",
    edition = "2021",
    deps = [
        ":bazel_analysis",
        ":bazel_bzlmod",
        ":bazel_cmdline",
        ":bazel_events",
        ":bazel_exec",
        ":bazel_options",
        ":bazel_packages",
        ":bazel_proto",
        ":bazel_query",
        ":bazel_remote",
        ":bazel_rules",
        ":bazel_sandbox",
        ":bazel_skyframe",
        ":bazel_starlark",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_server_test",
    crate = ":bazel_server",
)

rust_binary(
    name = "bazel-server-bin",
    srcs = ["bazel-server/src/main.rs"],
    edition = "2021",
    deps = [
        ":bazel_options",
        ":bazel_proto",
        ":bazel_server",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_library(
    name = "bazel_client_lib",
    srcs = glob(
        ["bazel-client/src/**/*.rs"],
        exclude = ["bazel-client/src/main.rs"],
    ),
    crate_name = "bazel_client",
    edition = "2021",
    deps = [
        ":bazel_cmdline",
        ":bazel_options",
        ":bazel_proto",
        ":bazel_server",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_client_test",
    crate = ":bazel_client_lib",
)

rust_binary(
    name = "bazel-rs",
    srcs = ["bazel-client/src/main.rs"],
    edition = "2021",
    deps = [
        ":bazel_client_lib",
        ":bazel_options",
        ":bazel_proto",
        ":bazel_server",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

copy_file(
    name = "bazel-rs-exe",
    src = ":bazel-rs",
    out = "bazel-rs.exe",
    allow_symlink = True,
    is_executable = True,
)

rust_shared_library(
    name = "bazel_jni_bridge",
    srcs = glob(["bazel-jni-bridge/src/**/*.rs"]),
    crate_name = "bazel_jni_bridge",
    edition = "2021",
    deps = [
        ":bazel_cmdline",
        ":bazel_skyframe",
        ":bazel_util",
        ":bazel_vfs",
    ],
)

rust_test(
    name = "bazel_jni_bridge_test",
    crate = ":bazel_jni_bridge",
)
