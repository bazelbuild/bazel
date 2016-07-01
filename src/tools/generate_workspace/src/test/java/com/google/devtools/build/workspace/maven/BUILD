filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/generate_workspace/src/test/java/com/google/devtools/build/workspace:__pkg__"],
)

java_test(
    name = "RuleTest",
    srcs = ["RuleTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven:rule",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ResolverTest",
    srcs = ["ResolverTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven:rule",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)
