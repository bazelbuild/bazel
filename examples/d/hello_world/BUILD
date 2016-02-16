package(default_visibility = ["//visibility:public"])

load("//tools/build_defs/d:d.bzl", "d_binary", "d_docs")

d_binary(
    name = "hello_world",
    srcs = ["hello_world.d"],
    deps = [
        "//examples/d/hello_lib:greeter",
        "//examples/d/hello_lib:native_greeter",
    ],
)

d_docs(
    name = "hello_world_docs",
    dep = ":hello_world",
)
