package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_test(
    name = "StringUnsafeTest",
    srcs = ["StringUnsafeTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
