package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools:__pkg__"],
)

config_setting(
    name = "darwin",
    values = {"host_cpu": "darwin"},
)

config_setting(
    name = "k8",
    values = {"host_cpu": "k8"},
)

filegroup(
    name = "rustc",
    srcs = select({
        ":darwin": ["@rust_darwin_x86_64//:rustc"],
        ":k8": ["@rust_linux_x86_64//:rustc"],
    }),
)

filegroup(
    name = "rustc_lib",
    srcs = select({
        ":darwin": ["@rust_darwin_x86_64//:rustc_lib"],
        ":k8": ["@rust_linux_x86_64//:rustc_lib"],
    }),
)

filegroup(
    name = "rustdoc",
    srcs = select({
        ":darwin": ["@rust_darwin_x86_64//:rustdoc"],
        ":k8": ["@rust_linux_x86_64//:rustdoc"],
    }),
)

filegroup(
    name = "rustlib",
    srcs = select({
        ":darwin": ["@rust_darwin_x86_64//:rustlib"],
        ":k8": ["@rust_linux_x86_64//:rustlib"],
    }),
)
