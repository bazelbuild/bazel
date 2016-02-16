package(default_visibility = ["//visibility:public"])

load(
    "//tools/build_defs/d:d.bzl",
    "d_docs",
    "d_library",
    "d_source_library",
    "d_test",
)

d_library(
    name = "greeter",
    srcs = ["greeter.d"],
)

d_test(
    name = "greeter_test",
    srcs = ["greeter_test.d"],
    deps = [":greeter"],
)

cc_library(
    name = "native_greeter_lib",
    srcs = ["native-greeter.c"],
    hdrs = ["native-greeter.h"],
)

d_source_library(
    name = "native_greeter",
    srcs = ["native_greeter.d"],
    deps = [":native_greeter_lib"],
)

d_docs(
    name = "greeter_docs",
    dep = ":greeter",
)

d_docs(
    name = "native_greeter_docs",
    dep = ":native_greeter",
)
