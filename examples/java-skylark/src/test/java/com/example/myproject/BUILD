load("/tools/build_rules/java_rules_skylark", "java_test")

java_test(
    name = "pass",
    srcs = ["TestHello.java"],
    deps = [
        "//examples/java-skylark/src/main/java/com/example/myproject:hello-lib",
        "//third_party:junit4-skylark",
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
