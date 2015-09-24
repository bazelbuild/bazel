package(default_visibility = ["//visibility:public"])

load("/tools/build_rules/rust/rust", "rust_library", "rust_docs", "rust_test")

rust_library(
    name = "hello_lib",
    srcs = [
        "src/greeter.rs",
        "src/lib.rs",
    ],
)

rust_docs(
    name = "hello_lib_docs",
    dep = ":hello_lib",
)

rust_test(
    name = "greeting",
    srcs = ["tests/greeting.rs"],
    deps = [":hello_lib"],
)
