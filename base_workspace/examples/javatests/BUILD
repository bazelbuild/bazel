java_test(
    name = "hello",
    srcs = ["test/TestHello.java"],
    deps = [
        "//examples/java:hello-lib",
        "//third_party:junit4",
    ],
)

java_test(
    name = "bye",
    srcs = ["test/TestBye.java"],
    deps = [
        "//examples/java:bye-lib",
        "//third_party:junit4",
    ],
)

java_test(
    name = "fail",
    srcs = ["test/Fail.java"],
    deps = ["//third_party:junit4"],
)

java_test(
    name = "should_fail",
    srcs = glob(["test/Test*.java"]),
    deps = [
        "//examples/java:hello-lib",
        "//third_party:junit4",
    ],
)
