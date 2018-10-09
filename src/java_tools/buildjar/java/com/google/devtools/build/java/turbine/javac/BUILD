package(default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"])

java_library(
    name = "javac_turbine",
    srcs = ["JavacTurbine.java"],
    deps = [
        ":formatted_diagnostic",
        ":javac_transitive",
        ":javac_turbine_compile_request",
        ":javac_turbine_compile_result",
        ":javac_turbine_compiler",
        ":zip_util",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:javac_options",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:turbine",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "javac_turbine_compile_request",
    srcs = ["JavacTurbineCompileRequest.java"],
    deps = [
        ":formatted_diagnostic",
        ":javac_transitive",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "javac_turbine_compile_result",
    srcs = ["JavacTurbineCompileResult.java"],
    deps = [
        ":formatted_diagnostic",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "javac_turbine_compiler",
    srcs = ["JavacTurbineCompiler.java"],
    deps = [
        ":formatted_diagnostic",
        ":javac_transitive",
        ":javac_turbine_compile_request",
        ":javac_turbine_compile_result",
        ":javac_turbine_java_compiler",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/statistics",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "javac_turbine_java_compiler",
    srcs = ["JavacTurbineJavaCompiler.java"],
    deps = [
        ":javac_transitive",
        ":tree_pruner",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/statistics",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "javac_transitive",
    srcs = ["JavacTransitive.java"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:turbine",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "zip_util",
    srcs = ["ZipUtil.java"],
    deps = [
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "tree_pruner",
    srcs = ["TreePruner.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "formatted_diagnostic",
    srcs = ["FormattedDiagnostic.java"],
    deps = [
        "//third_party/java/jdk/langtools:javac",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + [
        "BUILD",
    ],
)
