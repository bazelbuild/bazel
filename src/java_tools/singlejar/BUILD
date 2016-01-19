package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/java_tools/singlejar/java/com/google/devtools/build/zip:srcs"],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

java_library(
    name = "libSingleJar",
    srcs = glob(["java/**/singlejar/**/*.java"]),
    visibility = [
        "//src:__subpackages__",
        "//third_party/java/apkbuilder:__pkg__",
    ],
    deps = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/main/java/com/google/devtools/build/lib:shell",
        "//third_party:jsr305",
    ],
)

java_binary(
    name = "SingleJar",
    main_class = "com.google.devtools.build.singlejar.SingleJar",
    runtime_deps = [":libSingleJar"],
)

java_test(
    name = "tests",
    srcs = glob(["javatests/**/singlejar/**/*.java"]),
    test_class = "com.google.devtools.build.singlejar.SingleJarTests",
    deps = [
        ":libSingleJar",
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/main/java/com/google/devtools/build/lib:shell",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "zipTests",
    size = "enormous",
    srcs = glob(["javatests/**/zip/**/*.java"]),
    tags = ["zip"],
    test_class = "com.google.devtools.build.zip.ZipTests",
    deps = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

#
# Bootstrapping using Skylark rules
#
load("/tools/build_rules/java_rules_skylark", "bootstrap_java_library", "bootstrap_java_binary")

bootstrap_java_library(
    name = "skylark-deps",
    jars = [
        "//third_party:jsr305-jars",
    ],
)

bootstrap_java_binary(
    name = "bootstrap",
    srcs = glob(["java/**/*.java"]) + ["//src/java_tools/singlejar/java/com/google/devtools/build/zip:java-srcs"],
    main_class = "com.google.devtools.build.singlejar.SingleJar",
    visibility = ["//visibility:public"],
    deps = [
        ":skylark-deps",
        "//src/main/java/com/google/devtools/build/lib:shell-skylark",
    ],
)
