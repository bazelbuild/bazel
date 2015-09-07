package(default_visibility = ["//visibility:public"])

load("/tools/build_defs/d/d", "d_binary")

d_binary(
    name = "hello_world",
    srcs = ["hello_world.d"],
    deps = [
        "//examples/d/hello_lib:greeter",
        "//examples/d/hello_lib:native_greeter",
    ],
)
