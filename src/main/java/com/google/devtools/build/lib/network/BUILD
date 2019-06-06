package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
)

java_library(
    name = "connectivity",
    srcs = ["ConnectivityModule.java"],
    deps = [
        ":connectivity_status",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "connectivity_status",
    srcs = [
        "ConnectivityStatus.java",
        "ConnectivityStatusProvider.java",
    ],
    deps = ["//third_party:guava"],
)
