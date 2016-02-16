package_group(
    name = "packages",
    packages = ["//src/java_tools/buildjar/..."],
)

package(default_visibility = [":packages"])

java_library(
    name = "javac_turbine",
    srcs = ["JavacTurbine.java"],
    deps = [
        ":javac_turbine_compile_request",
        ":javac_turbine_compile_result",
        ":javac_turbine_compiler",
        ":zip_output_file_manager",
        ":zip_util",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:turbine_options",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:turbine_options_parser",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "javac_turbine_compile_request",
    srcs = ["JavacTurbineCompileRequest.java"],
    deps = [
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
        ":zip_output_file_manager",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "javac_turbine_compiler",
    srcs = ["JavacTurbineCompiler.java"],
    deps = [
        ":javac_turbine_compile_request",
        ":javac_turbine_compile_result",
        ":javac_turbine_java_compiler",
        ":zip_output_file_manager",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "javac_turbine_java_compiler",
    srcs = ["JavacTurbineJavaCompiler.java"],
    deps = [
        ":tree_pruner",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "zip_output_file_manager",
    srcs = ["ZipOutputFileManager.java"],
    deps = [
        "//third_party:jsr305",
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
