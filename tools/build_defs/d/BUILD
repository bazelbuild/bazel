package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
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
    name = "dmd",
    srcs = select({
        ":darwin": ["@dmd_darwin_x86_64//:dmd"],
        ":k8": ["@dmd_linux_x86_64//:dmd"],
    }),
)

filegroup(
    name = "libphobos2",
    srcs = select({
        ":darwin": ["@dmd_darwin_x86_64//:libphobos2"],
        ":k8": ["@dmd_linux_x86_64//:libphobos2"],
    }),
)

filegroup(
    name = "phobos-src",
    srcs = select({
        ":darwin": ["@dmd_darwin_x86_64//:phobos-src"],
        ":k8": ["@dmd_linux_x86_64//:phobos-src"],
    }),
)

filegroup(
    name = "druntime-import-src",
    srcs = select({
        ":darwin": ["@dmd_darwin_x86_64//:druntime-import-src"],
        ":k8": ["@dmd_linux_x86_64//:druntime-import-src"],
    }),
)
