filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

cc_binary(
    name = "remote_execution_log_dump",
    srcs = ["remote_execution_log_dump.cpp"],
    deps = [
        "//src/main/protobuf:remote_execution_log_cc_proto",
    ],
)
