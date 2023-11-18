load("@rules_java//java:defs.bzl", "java_binary", "java_test")

package(default_testonly = 1)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/main/java/net/starlark/java:bazel"],
)

# EvalTests test the Starlark interpreter.
java_test(
    name = "EvalTests",
    srcs = [
        "EvalTests.java",  # (suite)
        "EvalUtilsTest.java",
        "EvaluationTest.java",
        "EvaluationTestCase.java",
        "Examples.java",
        "FunctionTest.java",
        "ImmutableKeyTrackingDictTest.java",
        "MethodLibraryTest.java",
        "MutabilityTest.java",
        "PrinterTest.java",
        "StarlarkAnnotationsTest.java",
        "StarlarkClassTest.java",
        "StarlarkEvaluationTest.java",
        "StarlarkFlagGuardingTest.java",
        "StarlarkListTest.java",
        "StarlarkMutableTest.java",
        "StarlarkThreadDebuggingTest.java",
        "StarlarkThreadTest.java",
    ],
    jvm_flags = ["-Dfile.encoding=UTF8"],
    deps = [
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

# Script-based tests of the Starlark interpreter.
java_test(
    name = "ScriptTest",
    srcs = ["ScriptTest.java"],
    data = glob(["testdata/*.star"]),
    jvm_flags = ["-Dfile.encoding=UTF8"],
    use_testrunner = False,
    deps = [
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/lib/json",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)

# Script-based benchmarks of the Starlark interpreter.
java_binary(
    name = "Benchmarks",
    srcs = ["Benchmarks.java"],
    data = glob(["testdata/bench_*.star"]),
    jvm_flags = ["-Dfile.encoding=UTF8"],
    deps = [
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/lib/json",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:guava",
    ],
)
