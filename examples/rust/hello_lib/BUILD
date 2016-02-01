package(default_visibility = ["//visibility:public"])

load(
    "//tools/build_rules/rust:rust.bzl",
    "rust_library",
    "rust_test",
    "rust_doc",
    "rust_doc_test",
)

rust_library(
    name = "hello_lib",
    srcs = [
        "src/greeter.rs",
        "src/lib.rs",
    ],
)

rust_test(
    name = "hello_lib_test",
    deps = [":hello_lib"],
)

rust_test(
    name = "greeting_test",
    srcs = ["tests/greeting.rs"],
    deps = [":hello_lib"],
)

rust_doc(
    name = "hello_lib_doc",
    dep = ":hello_lib",
)

rust_doc_test(
    name = "hello_lib_doc_test",
    dep = ":hello_lib",
)
