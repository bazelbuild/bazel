package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src/tools/remote:__pkg__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/tools/remote/src/test/java/com/google/devtools/build/remote/worker/http:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)
