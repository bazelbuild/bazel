package(default_visibility = ["//visibility:public"])

load("//tools/build_rules/rust:rust.bzl", "rust_binary", "rust_doc", "rust_doc_test")

rust_binary(
    name = "hello_world",
    srcs = ["src/main.rs"],
    deps = ["//examples/rust/hello_lib"],
)

rust_doc(
    name = "hello_world_doc",
    dep = ":hello_world",
)

rust_doc_test(
    name = "hello_world_doc_test",
    dep = ":hello_world",
)
