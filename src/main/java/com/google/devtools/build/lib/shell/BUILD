package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

# Library for dealing with executable commands, including their arguments and runtime environment
# (environment variables, working directory). It lets a caller execute a command, get its results,
# and optionally forward interrupts to the subprocess. The library also handles creating threads to
# ensure timely reading of subprocess outputs.
java_library(
    name = "shell",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

# Bootstrapping SingleJar using Skylark rules
#
load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library")

bootstrap_java_library(
    name = "shell-skylark",
    srcs = glob(["*.java"]),
    jars = [
        "//third_party:auto_value-jars",
        "//third_party:bootstrap_guava_and_error_prone-jars",
    ],
)
