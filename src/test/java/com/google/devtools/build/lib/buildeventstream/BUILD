package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

java_test(
    name = "google",
    srcs = glob(["*.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = ["//src/test/java/com/google/devtools/build/lib:test_runner"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/authentication/google",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)
