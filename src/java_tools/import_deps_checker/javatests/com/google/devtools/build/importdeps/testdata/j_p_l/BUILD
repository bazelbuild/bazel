load("@com_google_protobuf//bazel:java_proto_library.bzl", "java_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = False,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "a",
    srcs = ["A.java"],
    deps = [
        ":b",
        ":p1_java_proto",
        ":p2_java_proto",
    ],
)

java_library(
    name = "b",
    srcs = ["B.java"],
    deps = [
        ":p1_java_proto",
    ],
)

java_proto_library(
    name = "p1_java_proto",
    deps = [":p1"],
)

proto_library(
    name = "p1",
    srcs = ["p1.proto"],
    deps = ["p2"],
)

java_proto_library(
    name = "p2_java_proto",
    deps = [":p2"],
)

proto_library(
    name = "p2",
    srcs = ["p2.proto"],
)
