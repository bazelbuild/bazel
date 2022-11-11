load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
        exclude = [
            ".*",
            "*~",
        ],  # .swp files and such
    ),
    visibility = ["//tools/java/runfiles:__pkg__"],
)

java_test(
    name = "RunfilesTest",
    srcs = ["RunfilesTest.java"],
    test_class = "com.google.devtools.build.runfiles.RunfilesTest",
    deps = [
        ":test_deps",
    ],
)

java_test(
    name = "UtilTest",
    srcs = ["UtilTest.java"],
    test_class = "com.google.devtools.build.runfiles.UtilTest",
    deps = [
        ":test_deps",
    ],
)

java_library(
    name = "test_deps",
    testonly = 1,
    exports = [
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
        "//tools/java/runfiles",
    ],
)
