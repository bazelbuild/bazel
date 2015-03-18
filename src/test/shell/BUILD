package(default_visibility = ["//visibility:private"])

sh_library(
    name = "bashunit",
    srcs = ["unittest.bash"],
    data = [
        "testenv.sh",
        "//tools:test_sharding_compliant",
    ],
    visibility = ["//visibility:public"],
)

sh_test(
    name = "bashunit_test",
    srcs = ["unittest_test.sh"],
    data = [":bashunit"],
)
