package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
)

java_test(
    name = "MemoizingInterruptibleSupplierTest",
    size = "small",
    srcs = ["MemoizingInterruptibleSupplierTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/supplier",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
