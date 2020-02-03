load("@rules_java//java:defs.bzl", "java_test")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_test(
    name = "GnuVersionParserTest",
    size = "small",
    srcs = ["GnuVersionParserTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/versioning",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "SemVerTest",
    size = "small",
    srcs = ["SemVerTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/versioning",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
