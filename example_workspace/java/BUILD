load("rules/java_library_oss", "java_library")

load("rules/java_binary_oss", "java_binary")

java_library(
    name = "hello-lib",
    srcs = ["Greeter.java"],
)

java_binary(
    name = "hello-world",
    main_class = "test.Greeter",
    deps = [":hello-lib"],
)
