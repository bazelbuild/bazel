package_group(
    name = "packages",
    packages = [
        "//src/java_tools/buildjar/...",
    ],
)

package(default_visibility = [":packages"])

# Exports the canonical javac bootclasspath artifact locations
genrule(
    name = "javac-bootclasspath-locations",
    srcs = ["@bazel_tools//tools/jdk:bootclasspath"],
    outs = ["JavacBootclasspathLocations.java"],
    cmd = "$(location javac-bootclasspath-locations.sh) $@ $(GENDIR) '' $(SRCS)",
    tools = ["javac-bootclasspath-locations.sh"],
)

# Provides java-level access to the canonical javac bootclasspath
java_library(
    name = "javac-bootclasspath-locations-lib",
    srcs = [":javac-bootclasspath-locations"],
)

java_library(
    name = "JavacBootclasspath",
    srcs = ["JavacBootclasspath.java"],
    data = ["@bazel_tools//tools/jdk:bootclasspath"],
    visibility = ["//visibility:public"],
    deps = [":javac-bootclasspath-locations-lib"],
)

# Exports the canonical javac jar location
genrule(
    name = "java-langtools-location",
    srcs = ["//third_party/java/jdk/langtools:javac_jar-9-dev-r4023-1"],
    outs = ["JavaLangtoolsLocation.java"],
    cmd = "$(location java-langtools-location.sh) $@ $(GENDIR) io_bazel/ $(SRCS)",
    tools = ["java-langtools-location.sh"],
)

# Provides java-level access to the canonical javac jar
java_library(
    name = "java-langtools-location-lib",
    srcs = [":java-langtools-location"],
    visibility = ["//visibility:private"],
)

java_library(
    name = "JavaLangtools",
    srcs = ["JavaLangtools.java"],
    data = ["//third_party/java/jdk/langtools:javac_jar-9-dev-r4023-1"],
    visibility = ["//visibility:public"],
    deps = [":java-langtools-location-lib"],
)

# Exports javacopts from the current java toolchain
java_binary(
    name = "JavaBuilderConfigGenerator",
    srcs = ["JavaBuilderConfigGenerator.java"],
    main_class = "com.google.devtools.build.java.bazel.JavaBuilderConfigGenerator",
    deps = [
        "//src/main/java/com/google/devtools/build/lib:java-toolchain-parser",
        "//third_party:guava",
    ],
)

# Exports javacopts from the current java toolchain
genquery(
    name = "java_toolchain_content",
    expression = "kind(java_toolchain, deps(//tools/defaults:java_toolchain))",
    opts = ["--output=proto"],
    scope = ["//tools/defaults:java_toolchain"],
)

genrule(
    name = "javabuilder-javacopts",
    srcs = [":java_toolchain_content"],
    outs = ["JavaBuilderJavacOpts.java"],
    cmd = "$(location :JavaBuilderConfigGenerator) $< > $@",
    tools = [":JavaBuilderConfigGenerator"],
)

# Provides java-level access to the javacopts in the current java toolchain
java_library(
    name = "javabuilder-javacopts-lib",
    srcs = [":javabuilder-javacopts"],
)

java_library(
    name = "JavaBuilderConfig",
    srcs = ["JavaBuilderConfig.java"],
    deps = [
        ":javabuilder-javacopts-lib",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:javac_options",
        "//third_party:guava",
    ],
)

# Provides programmatic access to a bazel compatible javac.
# Use this instead of ToolProvider.getSystemJavaCompiler().
java_library(
    name = "BazelJavaCompiler",
    srcs = ["BazelJavaCompiler.java"],
    data = ["//third_party/java/jdk/langtools:javac_jar-9-dev-r4023-1"],
    visibility = [
        "//src:__subpackages__",
        "//third_party/ijar:__subpackages__",
    ],
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
    deps = [":BazelJavaCompiler"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/java_tools/buildjar:__pkg__"],
)
