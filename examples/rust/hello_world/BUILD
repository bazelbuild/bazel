package(default_visibility = ["//visibility:public"])

load("/tools/build_rules/rust/rust", "rust_binary", "rust_docs")

rust_binary(
    name = "hello_world",
    srcs = ["src/main.rs"],
    deps = ["//examples/rust/hello_lib"],
)

rust_docs(
    name = "hello_world_docs",
    dep = ":hello_world",
)
