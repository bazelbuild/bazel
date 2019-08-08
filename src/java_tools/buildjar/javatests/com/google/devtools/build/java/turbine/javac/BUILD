package(default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"])

java_library(
    name = "AbstractJavacTurbineCompilationTest",
    testonly = 1,
    srcs = ["AbstractJavacTurbineCompilationTest.java"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/bazel:JavacBootclasspath",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:javac_turbine",
        "//src/main/protobuf:deps_java_proto",
        "//third_party:asm",
        "//third_party:asm-util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party:turbine",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_test(
    name = "JavacTurbineTest",
    srcs = ["JavacTurbineTest.java"],
    data = [
        "//third_party/java/jdk/langtools:javac_jar",
    ],
    deps = [
        ":AbstractJavacTurbineCompilationTest",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:javac_turbine",
        "//src/main/protobuf:deps_java_proto",
        "//third_party:asm",
        "//third_party:asm-util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party:turbine",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_test(
    name = "ProcessorClasspathTest_bootclasspath",
    srcs = ["//src/java_tools/buildjar/javatests/com/google/devtools/build/java/turbine/javac:ProcessorClasspathTest.java"],
    data = ["//third_party/java/jdk/langtools:javac_jar"],
    test_class = "com.google.devtools.build.java.turbine.javac.ProcessorClasspathTest",
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:javac_turbine",
        "//src/java_tools/buildjar/javatests/com/google/devtools/build/java/turbine/javac:AbstractJavacTurbineCompilationTest",
        "//src/main/protobuf:deps_java_proto",
        "//third_party:asm",
        "//third_party:asm-util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party:turbine",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_test(
    name = "TreePrunerTest",
    srcs = ["TreePrunerTest.java"],
    data = [
        "//third_party/java/jdk/langtools:javac_jar",
    ],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:tree_pruner",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/java/jdk/langtools:javac",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["*.java"]) + [
        "BUILD",
    ],
)
