package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
)

java_library(
    name = "cache",
    srcs = ["RepositoryCache.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib:vfs"],
)
