load("rules/java_library_oss", "java_library")

load("rules/java_binary_oss", "java_binary", "java_test")

java_library(
    name = "hello-lib",
    srcs = ["Greeter.java"],
    data = ["greeting.txt"],
)

java_binary(
    name = "hello-world",
    main_class = "test.Greeter",
    deps = [":hello-lib"],
)

# Testing.

java_library(
    name = "fail-lib",
    srcs = ["Fail.java"],
)

java_test(
    name = "pass_test",
    main_class = "test.Greeter",
    deps = [":hello-lib"],
)

java_test(
    name = "fail_test",
    main_class = "test.Fail",
    deps = [":fail-lib"],
)
