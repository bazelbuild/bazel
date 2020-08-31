load("@rules_java//java:defs.bzl", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/syntax/util:srcs",
    ],
)

# frontend_test tests the Starlark parser/resolver.
java_test(
    name = "frontend_test",
    srcs = [
        "FileLocationsTest.java",
        "LValueBoundNamesTest.java",
        "LexerTest.java",
        "LocationTest.java",
        "NodePrinterTest.java",
        "NodeVisitorTest.java",
        "ParserInputTest.java",
        "ParserTest.java",
        "ResolverTest.java",
        "StarlarkFileTest.java",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

# evaluator_test tests the Starlark interpreter.
java_test(
    name = "evaluator_test",
    srcs = [
        # CpuProfilerTest.java doesn't run under OSS Bazel
        "EvalUtilsTest.java",
        "EvaluationTest.java",
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
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/net/starlark/java/annot",
        "//src/test/java/com/google/devtools/build/lib/syntax/util",  # TODO(adonovan): break this dep
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
)
