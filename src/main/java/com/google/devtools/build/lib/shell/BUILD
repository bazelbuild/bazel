load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library")
load("@rules_java//java:defs.bzl", "java_library")

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
        "//src/main/java/com/google/devtools/build/lib/jni",
        "//src/main/java/com/google/devtools/build/lib/util:describable_execution_unit",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/windows:processes",
        "//src/main/protobuf:execution_statistics_java_proto",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
    ],
)

# Bootstrapping SingleJar using Starlark rules
#

bootstrap_java_library(
    name = "shell-starlark",
    srcs = glob(
        ["*.java"],
        exclude = ["ExecutionStatistics.java"],
    ),
    jars = [
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/jni",
        "//src/main/java/com/google/devtools/build/lib/util:describable_execution_unit",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/windows:processes",
        "//third_party:auto_value-jars",
        "//third_party:flogger-jars",
        "//third_party:bootstrap_guava_and_error_prone-jars",
    ],
)
