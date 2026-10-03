load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = [
        "//javatests/com/google/devtools/build/lib:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib:__subpackages__",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "ExecGraphParserTest",
    size = "small",
    srcs = ["ExecGraphParserTest.java"],
    test_class = "com.google.devtools.build.execgraph.ExecGraphParserTest",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util/io:io-proto",
        "//src/main/protobuf:execution_graph_java_proto",
        "//src/tools/execgraph/src/main/java/com/google/devtools/build/execgraph:parser",
        "//third_party:junit4",
        "//third_party:truth",
        "@zstd-jni",
    ],
)
