package(default_visibility = ["//visibility:public"])

java_library(
    name = "bye-lib",
    srcs = ["test/Greeter.java"],
    resources = ["greeting.txt"],
)

java_library(
    name = "hello-lib",
    srcs = ["test/Greeter.java"],
)

java_binary(
    name = "bye-world",
    main_class = "test.Greeter",
    runtime_deps = [":bye-lib"],
)

java_binary(
    name = "hello-world",
    main_class = "test.Greeter",
    runtime_deps = [":hello-lib"],
)

