package(default_visibility = ["//visibility:public"])

java_library(
    name = "statistics",
    srcs = ["BlazeJavacStatistics.java"],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party/java/jdk/langtools:javac",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + [
        "BUILD",
    ],
)
