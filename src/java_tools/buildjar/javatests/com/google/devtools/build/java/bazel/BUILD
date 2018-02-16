# Description:
#  Tests for tools for working with Java source code
package(default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"])

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
    test_class = "com.google.devtools.build.java.bazel.BazelJavaCompilerTest",
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/bazel:BazelJavaCompiler",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "BazelJavaCompilerTest.java",
    ],
)

test_suite(name = "all_tests")
