package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

java_test(
    name = "SkydocTest",
    size = "medium",
    srcs = ["SkydocTest.java"],
    shard_count = 1,
    visibility = ["//devtools/blaze/main:__pkg__"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skydoc:skydoc_lib",
        "//src/main/java/com/google/devtools/build/skydoc/rendering",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/test/java/com/google/devtools/build/lib/skylark:testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

# TODO(cparsons): Make test cases and golden-genrules into a macro.
sh_test(
    name = "simple_skydoc_e2e_test",
    srcs = ["skydoc_e2e_test_runner.sh"],
    args = [
        "$(location //src/main/java/com/google/devtools/build/skydoc:skydoc)",
        "$(location simple_test/input.txt)",
        "$(location simple_test/golden.txt)",
    ],
    data = [
        "simple_test/golden.txt",
        "simple_test/input.txt",
        "//src/main/java/com/google/devtools/build/skydoc",
    ],
)

genrule(
    name = "regenerate_simple_golden",
    srcs = [
        "simple_test/input.txt",
    ],
    outs = ["simple_output.txt"],
    cmd = "$(location //src/main/java/com/google/devtools/build/skydoc:skydoc) " +
          "$(location simple_test/input.txt) $(location simple_output.txt)",
    tools = ["//src/main/java/com/google/devtools/build/skydoc"],
)

sh_test(
    name = "unknown_name_skydoc_e2e_test",
    srcs = ["skydoc_e2e_test_runner.sh"],
    args = [
        "$(location //src/main/java/com/google/devtools/build/skydoc:skydoc)",
        "$(location unknown_name_test/input.txt)",
        "$(location unknown_name_test/golden.txt)",
    ],
    data = [
        "unknown_name_test/golden.txt",
        "unknown_name_test/input.txt",
        "//src/main/java/com/google/devtools/build/skydoc",
    ],
)

genrule(
    name = "regenerate_unknown_name_golden",
    srcs = [
        "unknown_name_test/input.txt",
    ],
    outs = ["unknown_name_output.txt"],
    cmd = "$(location //src/main/java/com/google/devtools/build/skydoc:skydoc) " +
          "$(location unknown_name_test/input.txt) $(location unknown_name_output.txt)",
    tools = ["//src/main/java/com/google/devtools/build/skydoc"],
)
