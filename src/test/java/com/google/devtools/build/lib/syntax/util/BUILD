package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "util",
    testonly = 1,
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:guava",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        # !!
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages:starlark_semantics_options",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
