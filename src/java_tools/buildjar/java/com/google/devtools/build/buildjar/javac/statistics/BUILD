package(default_visibility = ["//visibility:public"])

java_library(
    name = "statistics",
    srcs = ["BlazeJavacStatistics.java"],
    deps = [
        "any_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party/java/jdk/langtools:javac",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + [
        "BUILD",
    ],
)

# Depend on any_proto for bazel usage (the external repo doesn't expose any_java_proto, just
# the protos)
java_proto_library(
    name = "any_java_proto",
    deps = ["@com_google_protobuf//:any_proto"],
)
