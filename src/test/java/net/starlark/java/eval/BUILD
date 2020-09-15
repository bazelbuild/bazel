load("@rules_java//java:defs.bzl", "java_test")

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
        "MethodLibraryTest.java",
        "MutabilityTest.java",
        "PrinterTest.java",
        "StarlarkEvaluationTest.java",
        "StarlarkFlagGuardingTest.java",
        "StarlarkInterfaceUtilsTest.java",
        "StarlarkListTest.java",
        "StarlarkMutableTest.java",
        "StarlarkThreadDebuggingTest.java",
        "StarlarkThreadTest.java",
        "StringModuleTest.java",
    ],
    deps = [
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

# Script-based tests of the Starlark interpreter.
java_test(
    name = "ScriptTest",
    srcs = ["ScriptTest.java"],
    data = [
        "testdata/all_any.sky",
        "testdata/and_or_not.sky",
        "testdata/comprehension.sky",
        "testdata/cycles.sky",
        "testdata/dict.sky",
        "testdata/equality.sky",
        "testdata/function.sky",
        "testdata/int.sky",
        "testdata/int_constructor.sky",
        "testdata/int_function.sky",
        "testdata/list_mutation.sky",
        "testdata/list_slices.sky",
        "testdata/min_max.sky",
        "testdata/range.sky",
        "testdata/reversed.sky",
        "testdata/sorted.sky",
        "testdata/string_elems.sky",
        "testdata/string_find.sky",
        "testdata/string_format.sky",
        "testdata/string_misc.sky",
        "testdata/string_partition.sky",
        "testdata/string_slice_index.sky",
        "testdata/string_split.sky",
        "testdata/string_splitlines.sky",
        "testdata/string_test_characters.sky",
    ],
    use_testrunner = False,
    deps = [
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:guava",
    ],
)
