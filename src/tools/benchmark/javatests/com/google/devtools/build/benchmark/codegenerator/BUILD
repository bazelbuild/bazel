package(default_visibility = ["//src/tools/benchmark:__subpackages__"])

java_test(
    name = "JavaCodeGeneratorHelperTest",
    srcs = ["JavaCodeGeneratorHelperTest.java"],
    deps = [
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark/codegenerator:codegenerator_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "JavaCodeGeneratorTest",
    srcs = ["JavaCodeGeneratorTest.java"],
    deps = [
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark/codegenerator:codegenerator_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "CppCodeGeneratorHelperTest",
    srcs = ["CppCodeGeneratorHelperTest.java"],
    deps = [
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark/codegenerator:codegenerator_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "CppCodeGeneratorTest",
    srcs = ["CppCodeGeneratorTest.java"],
    deps = [
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark/codegenerator:codegenerator_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "MainTest",
    srcs = ["MainTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark/codegenerator:codegenerator_lib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
