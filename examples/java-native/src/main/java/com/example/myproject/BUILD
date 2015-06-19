package(default_visibility = ["//visibility:public"])

java_binary(
    name = "hello-world",
    main_class = "com.example.myproject.Greeter",
    runtime_deps = [":hello-lib"],
)

java_library(
    name = "hello-lib",
    srcs = glob(
        ["*.java"],
        exclude = ["HelloErrorProne.java"],
    ),
)

java_binary(
    name = "hello-resources",
    main_class = "com.example.myproject.Greeter",
    runtime_deps = [":custom-greeting"],
)

java_library(
    name = "custom-greeting",
    srcs = ["Greeter.java"],
    resources = ["//examples/java-native/src/main/resources:greeting"],
)

java_library(
    name = "hello-error-prone",
    srcs = ["HelloErrorProne.java"],
)

filegroup(
    name = "srcs",
    srcs = ["BUILD"] + glob(["**/*.java"]),
)
