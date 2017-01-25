package(default_visibility = ["//src/tools/benchmark:__subpackages__"])

java_binary(
    name = "codegenerator",
    srcs = glob(["*.java"]),
    main_class = "com.google.devtools.build.benchmark.codegenerator.Main",
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party/java/javapoet",
    ],
)

java_library(
    name = "codegenerator_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party/java/javapoet",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
