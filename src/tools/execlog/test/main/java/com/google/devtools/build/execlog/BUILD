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
    name = "ExecLogParserTest",
    size = "small",
    srcs = ["ExecLogParserTest.java"],
    test_class = "com.google.devtools.build.execlog.ExecLogParserTest",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_log_context_utils",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/protobuf:spawn_java_proto",
        "//src/tools/execlog/src/main/java/com/google/devtools/build/execlog:parser",
        "//third_party:junit4",
        "//third_party:truth",
        "@zstd-jni",
    ],
)
