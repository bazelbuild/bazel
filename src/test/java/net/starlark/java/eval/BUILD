load("@rules_java//java:defs.bzl", "java_test")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# Script-based tests of the Starlark interpreter.
java_test(
    name = "EvalTest",
    srcs = ["EvalTest.java"],
    data = [
        "testdata/all_any.sky",
        "testdata/and_or_not.sky",
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
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/net/starlark/java/annot",
        "//third_party:guava",
    ],
)
