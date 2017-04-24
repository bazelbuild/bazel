filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/java_tools/buildjar/javatests/com/google/devtools/build/buildjar:__pkg__"],
)

java_test(
    name = "ResourceJarBuilderTest",
    srcs = ["ResourceJarBuilderTest.java"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/resourcejar",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/java/jdk/langtools:javac-9-dev-r4023-1",
    ],
)
