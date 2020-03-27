package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]) + [
        "//src/test/java/com/google/devtools/build/lib/query2/aquery:srcs",
        "//src/test/java/com/google/devtools/build/lib/query2/cquery:srcs",
        "//src/test/java/com/google/devtools/build/lib/query2/engine:srcs",
        "//src/test/java/com/google/devtools/build/lib/query2/query/output:srcs",
        "//src/test/java/com/google/devtools/build/lib/query2/testutil:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

test_suite(
    name = "AllTests",
    tests = [
        "//src/test/java/com/google/devtools/build/lib/query2/aquery:ActionGraphProtoOutputFormatterCallbackTest",
        "//src/test/java/com/google/devtools/build/lib/query2/cquery:BuildOutputFormatterCallbackTest",
        "//src/test/java/com/google/devtools/build/lib/query2/cquery:ProtoOutputFormatterCallbackTest",
        "//src/test/java/com/google/devtools/build/lib/query2/cquery:TransitionsOutputFormatterTest",
    ],
)
