load("tools/build_rules/java_rules_oss", "java_library", "java_binary", "java_test")

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

java_binary(
    name = "hello-main",
    srcs = ["Greeter.java"],
    main_class = "test.Greeter",
)

# Testing.

java_test(
    name = "pass_test",
    main_class = "test.Greeter",
    deps = [":hello-lib"],
)

java_test(
    name = "fail_test",
    srcs = ["Fail.java"],
    main_class = "test.Fail",
)
