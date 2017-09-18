package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

# cc_toolchain now implicitly depends on @bazel_tools//tools/def_parser:def_parser
# We need to make sure @bazel_tools//tools/def_parser:def_parser is not
# a cc_binary, because otherwise, we'll introduce a cycle in dependency graph:
# .-> @bazel_tools//tools/def_parser:def_parser (cc_binary)
# |   cc_toolchain
# `-- @bazel_tools//tools/def_parser:def_parser (cc_binary)

filegroup(
    name = "def_parser",
    srcs = ["no_op.bat"],
)
