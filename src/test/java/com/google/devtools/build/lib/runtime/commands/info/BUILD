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
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands/info",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "RemoteRequestedInfoItemHandlerTest",
    size = "small",
    srcs = ["RemoteRequestedInfoItemHandlerTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands:paths_to_replace_utils",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands/info",
        "//src/main/protobuf:command_server_java_proto",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//third_party:junit4",
        "//third_party:truth",
        "@com_google_protobuf//:protobuf_java",
    ],
)
