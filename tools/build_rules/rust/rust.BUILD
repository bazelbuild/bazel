config_setting(
    name = "darwin",
    values = {"host_cpu": "darwin"},
)

config_setting(
    name = "k8",
    values = {"host_cpu": "k8"},
)

filegroup(
    name = "rustc",
    srcs = select({
        ":darwin": ["rustc/bin/rustc"],
        ":k8": ["rustc/bin/rustc"],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "rustc_lib",
    srcs = select({
        ":darwin": glob(["rustc/lib/*.dylib"]),
        ":k8": glob(["rustc/lib/*.so"]),
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "rustdoc",
    srcs = select({
        ":darwin": ["rustc/bin/rustdoc"],
        ":k8": ["rustc/bin/rustdoc"],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "rustlib",
    srcs = select({
        ":darwin": glob([
            "rust-std-x86_64-apple-darwin/lib/rustlib/x86_64-apple-darwin/lib/*.rlib",
            "rust-std-x86_64-apple-darwin/lib/rustlib/x86_64-apple-darwin/lib/*.dylib",
            "rust-std-x86_64-apple-darwin/lib/rustlib/x86_64-apple-darwin/lib/*.a",
            "rustc/lib/rustlib/x86_64-apple-darwin/lib/*.rlib",
            "rustc/lib/rustlib/x86_64-apple-darwin/lib/*.dylib",
            "rustc/lib/rustlib/x86_64-apple-darwin/lib/*.a",
        ]),
        ":k8": glob([
            "rust-std-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib/*.rlib",
            "rust-std-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib/*.so",
            "rust-std-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib/*.a",
            "rustc/lib/rustlib/x86_64-unknown-linux-gnu/lib/*.rlib",
            "rustc/lib/rustlib/x86_64-unknown-linux-gnu/lib/*.so",
            "rustc/lib/rustlib/x86_64-unknown-linux-gnu/lib/*.a",
        ]),
    }),
    visibility = ["//visibility:public"],
)
