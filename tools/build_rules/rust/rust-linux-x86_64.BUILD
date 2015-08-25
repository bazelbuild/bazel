BASE_DIR = "rust-1.2.0-x86_64-unknown-linux-gnu/"

filegroup(
    name = "rustc",
    srcs = [BASE_DIR + "rustc/bin/rustc"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "rustdoc",
    srcs = [BASE_DIR + "rustc/bin/rustdoc"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "rustlib",
    srcs = glob([
        BASE_DIR + "rustc/lib/rustlib/x86_64-unknown-linux-gnu/lib/*.rlib",
        BASE_DIR + "rustc/lib/rustlib/x86_64-unknown-linux-gnu/lib/*.so",
        BASE_DIR + "rustc/lib/rustlib/x86_64-unknown-linux-gnu/lib/*.a",
    ]),
    visibility = ["//visibility:public"],
)
