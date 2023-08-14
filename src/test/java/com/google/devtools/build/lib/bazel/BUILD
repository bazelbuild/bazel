package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]) + [
        "//src/test/java/com/google/devtools/build/lib/bazel/bzlmod:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/debug:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/execlog:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)
