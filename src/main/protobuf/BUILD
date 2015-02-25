package(default_visibility = ["//visibility:public"])

load("/tools/build_rules/genproto", "genproto")

[genproto(
    name = "proto_" + proto_file,
    src = proto_file + ".proto",
) for proto_file in [
    "build",
    "deps",
    "crosstool_config",
    "extra_actions_base",
    "test_status",
    "bundlemerge",
    "xcodegen",
]]
