# Description:
#   Tools for working with Java source code, using the Sun javac API.
#
# This is not the source code for javac itself.

package(
    default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"],
)

load(":build_defs.bzl", "default_javacopts")

# Exports the canonical javac bootclasspath artifact locations
genrule(
    name = "javac-bootclasspath-locations",
    srcs = ["@bazel_tools//tools/jdk:bootclasspath"],
    outs = ["JavacBootclasspathLocations.java"],
    cmd = "$(location javac-bootclasspath-locations.sh) $@ $(GENDIR) $(SRCS)",
    tools = ["javac-bootclasspath-locations.sh"],
)

# Provides java-level access to the canonical javac bootclasspath
java_library(
    name = "JavacBootclasspath",
    srcs = [
        "JavacBootclasspath.java",
        ":javac-bootclasspath-locations",
    ],
    data = ["@bazel_tools//tools/jdk:bootclasspath"],
    visibility = ["//visibility:public"],
    deps = ["//third_party:guava"],
)

# Exports the canonical javac jar location
genrule(
    name = "java-langtools-location",
    srcs = ["//third_party/java/jdk/langtools:javac_jar"],
    outs = ["JavaLangtoolsLocation.java"],
    cmd = "$(location java-langtools-location.sh) $@ $(BINDIR) $(SRCS)",
    tools = ["java-langtools-location.sh"],
)

# Provides java-level access to the canonical javac jar
java_library(
    name = "JavaLangtools",
    srcs = [
        "JavaLangtools.java",
        ":java-langtools-location",
    ],
    data = ["//third_party/java/jdk/langtools:javac_jar"],
    visibility = ["//visibility:public"],
)

# Exports javacopts from the current java toolchain
default_javacopts(
    name = "gen_default_javacopts",
    out = "JavaBuilderJavacOpts.java",
    template = "JavaBuilderJavacOpts.java.template",
)

# Provides java-level access to the javacopts in the current java toolchain
java_library(
    name = "JavaBuilderConfig",
    srcs = [
        "JavaBuilderConfig.java",
        "JavaBuilderJavacOpts.java",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:javac_options",
        "//third_party:guava",
    ],
)

# Provides programmatic access to a bazel compatible javac.
# Use this instead of ToolProvider.getSystemJavaCompiler().
java_library(
    name = "BazelJavaCompiler",
    srcs = ["BazelJavaCompiler.java"],
    data = ["//third_party/java/jdk/langtools:javac_jar"],
    visibility = ["//visibility:public"],
    deps = [
        ":JavaBuilderConfig",
        ":JavaLangtools",
        ":JavacBootclasspath",
        "//third_party:guava",
    ],
)

# Command line version of BazelJavaCompiler, interface-compatible
# with the javac command, for use with ant, for example.
java_library(
    name = "BazelJavac",
    srcs = ["BazelJavac.java"],
    visibility = ["//visibility:public"],
    deps = [":BazelJavaCompiler"],
)

java_binary(
    name = "bazeljavac",
    # Use same stack size as javac and JavaBuilder.
    jvm_flags = ["-Xss4m"],
    main_class = "com.google.devtools.build.java.bazel.BazelJavac",
    visibility = ["//visibility:public"],
    runtime_deps = [":BazelJavac"],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "*.java",
        "*.sh",
    ]) + [
        "BUILD",
        "JavaBuilderJavacOpts.java.template",
        "build_defs.bzl",
    ],
)
