package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__pkg__"],
)

exports_files([
    "Cargo.lock",
    "Cargo.toml",
])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/rust/util:srcs",
    ],
)
