package(default_visibility = [
    "//src/java_tools/junitrunner:junitrunner_package_group",
])

java_library(
    name = "util",
    srcs = glob(["*.java"]),
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:jsr305",
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + ["BUILD"],
)
