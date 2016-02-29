# Detect our platform,
# mac os x
config_setting(
    name = "darwin",
    values = {"host_cpu": "darwin"},
)

# linux amd64
config_setting(
    name = "linux",
    values = {"host_cpu": "k8"},
)

config_setting(
    name = "debug",
    values = {"compilation_mode": "dbg"},
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools:__pkg__"],
)
