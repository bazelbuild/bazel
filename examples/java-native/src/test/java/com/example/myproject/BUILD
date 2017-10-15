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
    test_class = "com.example.myproject.Fail",
    deps = ["//third_party:junit4"],
)

# This exercises the "cpu:<num>" tag.
#
# Expected behavior:
# test0 and test1 run together then one of the other two alone, and then the remaining one alone.
java_test(
    name = "sleep0",
    srcs = ["TestSleep.java"],
    test_class = "com.example.myproject.TestSleep",
    deps = ["//third_party:junit4"],
)

java_test(
    name = "sleep1",
    srcs = ["TestSleep.java"],
    test_class = "com.example.myproject.TestSleep",
    deps = ["//third_party:junit4"],
)

java_test(
    name = "sleep2",
    srcs = ["TestSleep.java"],
    tags = ["cpu:20000"],
    test_class = "com.example.myproject.TestSleep",
    deps = ["//third_party:junit4"],
)

java_test(
    name = "sleep3",
    srcs = ["TestSleep.java"],
    tags = ["cpu:20000"],
    test_class = "com.example.myproject.TestSleep",
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
