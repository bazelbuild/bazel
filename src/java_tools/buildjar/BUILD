package(default_visibility = ["//src:__subpackages__"])

java_binary(
    name = "JavaBuilder",
    srcs = glob(
        ["java/com/google/devtools/build/buildjar/**/*.java"],
        exclude = ["java/com/google/devtools/build/buildjar/javac/testing/**"],
    ),
    main_class = "com.google.devtools.build.buildjar.BazelJavaBuilder",
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper",
        "//src/main/protobuf:deps_proto",
        "//src/main/protobuf:java_compilation_proto",
        "//src/main/protobuf:worker_protocol_proto",
        "//third_party:error_prone",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:protobuf",
        "//tools/jdk:langtools-neverlink",
    ],
)

# Exports $(locations //tools/defaults:javac_bootclasspath) to the java world.
genrule(
    name = "javac-bootclasspath-locations",
    srcs = ["//tools/defaults:javac_bootclasspath"],
    outs = ["java/com/google/devtools/build/java/bazel/JavacBootclasspathLocations.java"],
    cmd = """
declare -a paths=($(SRCS)) && paths=($${paths[@]#$(GENDIR)/}) &&
IFS=: &&
cat > $@ <<EOF
package com.google.devtools.build.java.bazel;
public class JavacBootclasspathLocations {
  public static final String BOOTCLASSPATH = "$${paths[*]}";
}
EOF
""",
)

# Provides java-level access to //tools/defaults:javac_bootclasspath
# as defined by blaze's --javac_bootclasspath flag
java_library(
    name = "javac-bootclasspath-locations-lib",
    srcs = [":javac-bootclasspath-locations"],
    visibility = ["//visibility:private"],
)

java_library(
    name = "JavacBootclasspath",
    srcs = ["java/com/google/devtools/build/java/bazel/JavacBootclasspath.java"],
    data = ["//tools/defaults:javac_bootclasspath"],
    visibility = ["//visibility:public"],
    deps = [":javac-bootclasspath-locations-lib"],
)

# Exports $(location //tools/defaults:java_langtools) to the java world.
genrule(
    name = "java-langtools-location",
    srcs = ["//tools/defaults:java_langtools"],
    outs = ["java/com/google/devtools/build/java/bazel/JavaLangtoolsLocation.java"],
    cmd = """
path=$(SRCS) && path=$${path#$(GENDIR)/} &&
cat > $@ <<EOF
package com.google.devtools.build.java.bazel;
public class JavaLangtoolsLocation {
  public static final String FILE = "$${path}";
}
EOF
""",
)

# Provides java-level access to //tools/defaults:java_langtools
# as defined by bazel's --java_langtools flag
java_library(
    name = "java-langtools-location-lib",
    srcs = [":java-langtools-location"],
    visibility = ["//visibility:private"],
)

java_library(
    name = "JavaLangtools",
    srcs = ["java/com/google/devtools/build/java/bazel/JavaLangtools.java"],
    data = ["//tools/defaults:java_langtools"],
    visibility = ["//visibility:public"],
    deps = [":java-langtools-location-lib"],
)

# Utility to export javacopts to the java world from the default javabuilder (a
# java_binary deploy jar).
java_binary(
    name = "JavaBuilderConfigGenerator",
    srcs = ["java/com/google/devtools/build/java/bazel/JavaBuilderConfigGenerator.java"],
    main_class = "com.google.devtools.build.java.bazel.JavaBuilderConfigGenerator",
    deps = [
        "//src/main/java/com/google/devtools/build/lib:java-toolchain-parser",
        "//third_party:guava",
    ],
)

# Exports the default javacopts to the java world from the default javabuilder
# (a java_binary deploy jar).
# We could put any java target as scope and in deps here.
genquery(
    name = "java_toolchain_content",
    expression = "kind(java_toolchain, deps(//tools/defaults:java_toolchain))",
    opts = ["--output=proto"],
    scope = ["//tools/defaults:java_toolchain"],
)

genrule(
    name = "javabuilder-javacopts",
    srcs = [":java_toolchain_content"],
    outs = ["java/com/google/devtools/build/java/bazel/JavaBuilderJavacOpts.java"],
    cmd = "$(location :JavaBuilderConfigGenerator) $< > $@",
    tools = [":JavaBuilderConfigGenerator"],
)

# Provides java-level access to the default javacopts in the current
# JavaBuilder release as defined by bazel's --javabuilder_top flag
java_library(
    name = "javabuilder-javacopts-lib",
    srcs = [":javabuilder-javacopts"],
    visibility = ["//visibility:private"],
)

java_library(
    name = "JavaBuilderConfig",
    srcs = ["java/com/google/devtools/build/java/bazel/JavaBuilderConfig.java"],
    data = ["//tools/defaults:javabuilder"],
    deps = [":javabuilder-javacopts-lib"],
)

# Provides programmatic access to a bazel compatible javac.
# Use this instead of ToolProvider.getSystemJavaCompiler().
java_library(
    name = "BazelJavaCompiler",
    srcs = ["java/com/google/devtools/build/java/bazel/BazelJavaCompiler.java"],
    data = ["//tools/defaults:java_langtools"],
    visibility = [
        "//src:__subpackages__",
        "//third_party/ijar:__subpackages__",
    ],
    deps = [
        ":JavaBuilderConfig",
        ":JavaLangtools",
        ":JavacBootclasspath",
    ],
)

# Command line version of BazelJavaCompiler, interface-compatible
# with the javac command, for use with ant, for example.
java_library(
    name = "BazelJavac",
    srcs = ["java/com/google/devtools/build/java/bazel/BazelJavac.java"],
    deps = [":BazelJavaCompiler"],
)

# For the -Xbootclasspath option of the java_test of buildjar
genrule(
    name = "langtools-test",
    testonly = 1,
    srcs = ["//tools/defaults:java_langtools"],
    outs = ["langtools-test.jar"],
    cmd = "cp $< $@",
)

java_test(
    name = "BazelJavaCompilerTest",
    size = "small",
    srcs = ["javatests/com/google/devtools/build/java/bazel/BazelJavaCompilerTest.java"],
    jvm_flags = [
        # Simulates how Bazel invokes JavaBuilder
        "-Xbootclasspath/p:$${TEST_SRCDIR}/src/java_tools/buildjar/langtools-test.jar",
    ],
    tags = ["jdk8"],
    test_class = "com.google.devtools.build.java.bazel.BazelJavaCompilerTest",
    deps = [
        ":BazelJavaCompiler",
        ":langtools-test",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

#
# Bootstrapping using Skylark rules
#
load("//tools:build_rules/java_rules_skylark.bzl", "bootstrap_java_library", "bootstrap_java_binary")

bootstrap_java_library(
    name = "skylark-deps",
    jars = [
        "//third_party:error_prone-jars",
        "//third_party:guava-jars",
        "//third_party:jsr305-jars",
        "//third_party:protobuf-jars",
    ],
    neverlink_jars = ["//tools/defaults:java_langtools"],
    srcjars = [
        "//src/main/protobuf:deps_proto_srcjar",
        "//src/main/protobuf:worker_protocol_proto_srcjar",
        "//src/main/protobuf:java_compilation_proto_srcjar",
    ],
)

bootstrap_java_binary(
    name = "bootstrap",
    srcs = glob(
        ["java/com/google/devtools/build/buildjar/**/*.java"],
        exclude = ["java/com/google/devtools/build/buildjar/javac/testing/**"],
    ) + [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:JarCreator.java",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:JarHelper.java",
    ],
    main_class = "com.google.devtools.build.buildjar.BazelJavaBuilder",
    visibility = ["//visibility:public"],
    deps = [":skylark-deps"],
)

bootstrap_java_binary(
    name = "bootstrap_genclass",
    srcs = glob(["java/com/google/devtools/build/buildjar/genclass/*.java"]),
    main_class = "com.google.devtools.build.buildjar.genclass.GenClass",
    visibility = ["//visibility:public"],
    deps = [":skylark-deps"],
)
