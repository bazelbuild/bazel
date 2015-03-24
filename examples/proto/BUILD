package(default_visibility = ["//visibility:public"])

load("/tools/build_rules/genproto", "genproto")

genproto(
    name = "test_proto",
    src = "test.proto",
)

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "test.proto",
    ],
)
