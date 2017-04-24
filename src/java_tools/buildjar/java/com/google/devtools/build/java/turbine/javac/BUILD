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
        ":zip_util",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:JarOwner",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:javac_options",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:turbine",
        "//third_party/java/jdk/langtools:javac-9-dev-r4023-1",
    ],
)

java_library(
    name = "javac_turbine_compile_request",
    srcs = ["JavacTurbineCompileRequest.java"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac-9-dev-r4023-1",
    ],
)

java_library(
    name = "javac_turbine_compile_result",
    srcs = ["JavacTurbineCompileResult.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac-9-dev-r4023-1",
    ],
)

java_library(
    name = "javac_turbine_compiler",
    srcs = ["JavacTurbineCompiler.java"],
    deps = [
        ":javac_turbine_compile_request",
        ":javac_turbine_compile_result",
        ":javac_turbine_java_compiler",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//third_party:guava",
        "//third_party:jimfs",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac-9-dev-r4023-1",
    ],
)

java_library(
    name = "javac_turbine_java_compiler",
    srcs = ["JavacTurbineJavaCompiler.java"],
    deps = [
        ":javac_turbine_compile_request",
        ":tree_pruner",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac-9-dev-r4023-1",
    ],
)

java_library(
    name = "zip_util",
    srcs = ["ZipUtil.java"],
    deps = [
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac-9-dev-r4023-1",
    ],
)

java_library(
    name = "tree_pruner",
    srcs = ["TreePruner.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac-9-dev-r4023-1",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:__pkg__"],
)
