load("@rules_java//java:defs.bzl", "java_test")

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "UniverseScopeTest",
    srcs = ["UniverseScopeTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/query2/common:universe-scope",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
