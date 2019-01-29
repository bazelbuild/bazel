package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "callcounts",
    srcs = ["Callcounts.java"],
    deps = [
        "//third_party:guava",
        "//third_party/pprof:profile_java_proto",
    ],
)

java_library(
    name = "callcounts_module",
    srcs = ["CallcountsModule.java"],
    deps = [
        ":callcounts",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)
