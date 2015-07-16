java_library(
    name = "AllTests",
    srcs = ["AllTests.java"],
    deps = [
        "//src/test/java:testutil",
        "//third_party:junit4",
    ],
)

java_library(
    name = "util",
    srcs = ["ProtoInputStream.java"],
    deps = [
        "//external:javax/servlet/api",
        "//third_party:protobuf",
    ],
)

java_test(
    name = "dash",
    srcs = [
        "DashRequestTest.java",
    ],
    runtime_deps = [":AllTests"],
    deps = [
        "@appengine-java//:jars",
        "@easymock//jar",
        ":util",
        "//src/main/protobuf:proto_dash",
        "//src/tools/dash/src/main/java:servlets",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
