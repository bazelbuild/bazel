java_test(
    name = "BazelJavaCompilerTest",
    size = "small",
    srcs = ["BazelJavaCompilerTest.java"],
    data = [
        "//third_party/java/jdk/langtools:javac_jar",
    ],
    jvm_flags = [
        # Simulates how Bazel invokes JavaBuilder
        "-Xbootclasspath/p:$(location //third_party/java/jdk/langtools:javac_jar)",
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
