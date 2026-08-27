load("@rules_java//java:java_test.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "StdoutInfoItemHandlerTest",
    size = "small",
    srcs = ["StdoutInfoItemHandlerTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_command_cluster",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands/info",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "JavaHomeInfoItemTest",
    size = "small",
    srcs = ["JavaHomeInfoItemTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/runtime/commands/info",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "RemoteRequestedInfoItemHandlerTest",
    size = "small",
    srcs = ["RemoteRequestedInfoItemHandlerTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_command_cluster",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands:paths_to_replace_utils",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands/info",
        "//src/main/protobuf:command_server_java_proto",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//third_party/java/junit",
        "//third_party/java/protobuf:protobuf_lite",
        "//third_party/java/truth",
    ],
)
