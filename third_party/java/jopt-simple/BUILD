package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "jopt-simple",
    srcs = glob(["src/main/**/*.java"]),
    resources = [
        "src/main/resources/joptsimple/ExceptionMessages.properties",
        "src/main/resources/joptsimple/HelpFormatterMessages.properties",
    ],
)
