load("//tools:build_rules/java_rules_skylark.bzl", "java_library", "java_binary", "java_test")

package(default_visibility = ["//visibility:public"])

java_binary(
    name = "hello-world",
    srcs = ["Greeter.java"],
    main_class = "com.example.myproject.Greeter",
)

java_binary(
    name = "hello-data",
    main_class = "com.example.myproject.Greeter",
    deps = [":hello-lib"],
)

java_library(
    name = "hello-lib",
    srcs = ["Greeter.java"],
    data = ["//examples/java-skylark/src/main/resources:greeting"],
)

filegroup(
    name = "srcs",
    srcs = ["BUILD"] + glob(["**/*.java"]),
)
