load("//tools/build_rules:java_rules_skylark.bzl", "java_library", "java_test")

java_library(
    name = "junit4",
    jars = ["//third_party:junit4-jars"],
)

java_test(
    name = "pass",
    srcs = ["TestHello.java"],
    deps = [
        ":junit4",
        "//examples/java-skylark/src/main/java/com/example/myproject:hello-lib",
    ],
)

java_test(
    name = "fail",
    srcs = ["Fail.java"],
    main_class = "com.example.myproject.Fail",
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + ["BUILD"],
    visibility = ["//examples/java-skylark:__pkg__"],
)
