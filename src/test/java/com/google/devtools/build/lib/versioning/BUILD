load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "GnuVersionParserTest",
    size = "small",
    srcs = ["GnuVersionParserTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/versioning",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
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
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
