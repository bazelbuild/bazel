filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/java_tools/buildjar:__pkg__"],
)

java_test(
    name = "BazelJavaCompilerTest",
    size = "small",
    srcs = ["BazelJavaCompilerTest.java"],
    data = [
        "//third_party/java/jdk/langtools:javac_jar-9-dev-r4023-1",
    ],
    jvm_flags = [
        # Simulates how Bazel invokes JavaBuilder
        "-Xbootclasspath/p:$(location //third_party/java/jdk/langtools:javac_jar-9-dev-r4023-1)",
    ],
    tags = ["jdk8"],
    test_class = "com.google.devtools.build.java.bazel.BazelJavaCompilerTest",
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/bazel:BazelJavaCompiler",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

test_suite(name = "all_tests")
