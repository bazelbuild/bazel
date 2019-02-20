# Tests for the Skylark linter

java_test(
    name = "SkylintTests",
    srcs = glob(["*.java"]),
    test_class = "com.google.devtools.skylark.skylint.SkylintTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/tools/skylark/java/com/google/devtools/skylark/common",
        "//src/tools/skylark/java/com/google/devtools/skylark/skylint:skylint_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
    visibility = ["//src:__pkg__"],
)
