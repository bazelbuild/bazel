package(default_visibility = ["//visibility:public"])

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
        ":darwin": ["@rust-darwin-x86_64//:rustc"],
        ":k8": ["@rust-linux-x86_64//:rustc"],
    }),
)

filegroup(
    name = "rustc_lib",
    srcs = select({
        ":darwin": ["@rust-darwin-x86_64//:rustc_lib"],
        ":k8": ["@rust-linux-x86_64//:rustc_lib"],
    }),
)

filegroup(
    name = "rustlib",
    srcs = select({
        ":darwin": ["@rust-darwin-x86_64//:rustlib"],
        ":k8": ["@rust-linux-x86_64//:rustlib"],
    }),
)
