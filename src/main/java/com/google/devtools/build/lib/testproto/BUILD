load(":testproto.bzl", "testproto")

proto_library(
    name = "test_proto",
    srcs = ["subdir/test.proto"],
)

java_library(
    name = "test_java",
    srcs = ["subdir/hello.java"],
)

testproto(name="testproto_sky", protos=[":test_proto"], javas=[":test_java"])
