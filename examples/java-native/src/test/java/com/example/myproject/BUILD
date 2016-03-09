java_test(
    name = "hello",
    srcs = ["TestHello.java"],
    test_class = "com.example.myproject.TestHello",
    deps = [
        "//examples/java-native/src/main/java/com/example/myproject:hello-lib",
        "//third_party:junit4",
    ],
)

java_test(
    name = "custom",
    srcs = ["TestCustomGreeting.java"],
    test_class = "com.example.myproject.TestCustomGreeting",
    deps = [
        "//examples/java-native/src/main/java/com/example/myproject:custom-greeting",
        "//third_party:junit4",
    ],
)

java_test(
    name = "custom_with_test_class",
    srcs = glob(["Test*.java"]),
    test_class = "com.example.myproject.TestCustomGreeting",
    deps = [
        "//examples/java-native/src/main/java/com/example/myproject:custom-greeting",
        "//third_party:junit4",
    ],
)

java_test(
    name = "fail",
    srcs = ["Fail.java"],
    deps = ["//third_party:junit4"],
)

# This attempts to run TestCustomGreeting.java without any resources, so fails.
java_test(
    name = "resource-fail",
    srcs = glob(["Test*.java"]),
    test_class = "com.example.myproject.TestCustomGreeting",
    deps = [
        "//examples/java-native/src/main/java/com/example/myproject:hello-lib",
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + ["BUILD"],
    visibility = ["//examples/java-native:__pkg__"],
)
