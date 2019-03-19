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
    name = "StableSortTest",
    size = "small",
    srcs = ["StableSortTest.java"],
    test_class = "com.google.devtools.build.lib.bazel.execlog.StableSortTest",
    deps = [
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib/bazel/execlog:stable_sort",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:spawn_java_proto",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
    ],
)
