load(":skydoc_test.bzl", "skydoc_test")

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

skydoc_test(
    name = "simple_test",
    golden_file = "simple_test/golden.txt",
    input_file = "simple_test/input.txt",
    skydoc = "//src/main/java/com/google/devtools/build/skydoc",
)

skydoc_test(
    name = "unknown_name",
    golden_file = "unknown_name_test/golden.txt",
    input_file = "unknown_name_test/input.txt",
    skydoc = "//src/main/java/com/google/devtools/build/skydoc",
)

skydoc_test(
    name = "multiple_rules_test",
    golden_file = "multiple_rules_test/golden.txt",
    input_file = "multiple_rules_test/input.txt",
    skydoc = "//src/main/java/com/google/devtools/build/skydoc",
)
