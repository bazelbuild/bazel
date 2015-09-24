RUST_VERSION = "1.3.0"
LINUX_BASE_DIR = "rust-%s-x86_64-unknown-linux-gnu/" % RUST_VERSION
DARWIN_BASE_DIR = "rust-%s-x86_64-apple-darwin/" % RUST_VERSION

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
        ":darwin": [DARWIN_BASE_DIR + "rustc/bin/rustc"],
        ":k8": [LINUX_BASE_DIR + "rustc/bin/rustc"],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "rustc_lib",
    srcs = select({
        ":darwin": glob([DARWIN_BASE_DIR + "rustc/lib/*.dylib"]),
        ":k8": glob([LINUX_BASE_DIR + "rustc/lib/*.so"]),
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "rustdoc",
    srcs = select({
        ":darwin": [DARWIN_BASE_DIR + "rustc/bin/rustdoc"],
        ":k8": [LINUX_BASE_DIR + "rustc/bin/rustdoc"],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "rustlib",
    srcs = select({
        ":darwin": glob([
            DARWIN_BASE_DIR + "rustc/lib/rustlib/x86_64-apple-darwin/lib/*.rlib",
            DARWIN_BASE_DIR + "rustc/lib/rustlib/x86_64-apple-darwin/lib/*.dylib",
            DARWIN_BASE_DIR + "rustc/lib/rustlib/x86_64-apple-darwin/lib/*.a",
        ]),
        ":k8": glob([
            LINUX_BASE_DIR + "rustc/lib/rustlib/x86_64-unknown-linux-gnu/lib/*.rlib",
            LINUX_BASE_DIR + "rustc/lib/rustlib/x86_64-unknown-linux-gnu/lib/*.so",
            LINUX_BASE_DIR + "rustc/lib/rustlib/x86_64-unknown-linux-gnu/lib/*.a",
        ]),
    }),
    visibility = ["//visibility:public"],
)
