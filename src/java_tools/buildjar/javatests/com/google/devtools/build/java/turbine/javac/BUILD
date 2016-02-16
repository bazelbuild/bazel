java_test(
    name = "JavacTurbineTest",
    srcs = ["JavacTurbineTest.java"],
    tags = ["jdk8"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:turbine_options",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:javac_turbine",
        "//src/main/protobuf:deps_proto",
        "//third_party:asm",
        "//third_party:asm-util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_test(
    name = "TreePrunerTest",
    srcs = ["TreePrunerTest.java"],
    tags = ["jdk8"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:tree_pruner",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/java/jdk/langtools:javac",
    ],
)
