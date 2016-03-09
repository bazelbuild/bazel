package_group(
    name = "packages",
    packages = [
        "//src/java_tools/buildjar/...",
    ],
)

package(default_visibility = [":packages"])

# Exports $(locations //tools/defaults:javac_bootclasspath) to the java world.
genrule(
    name = "javac-bootclasspath-locations",
    srcs = ["//tools/defaults:javac_bootclasspath"],
    outs = ["JavacBootclasspathLocations.java"],
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
)

java_library(
    name = "JavacBootclasspath",
    srcs = ["JavacBootclasspath.java"],
    data = ["//tools/defaults:javac_bootclasspath"],
    visibility = ["//visibility:public"],
    deps = [":javac-bootclasspath-locations-lib"],
)

# Exports $(location //tools/defaults:java_langtools) to the java world.
genrule(
    name = "java-langtools-location",
    srcs = ["//tools/defaults:java_langtools"],
    outs = ["JavaLangtoolsLocation.java"],
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
    srcs = ["JavaLangtools.java"],
    data = ["//tools/defaults:java_langtools"],
    visibility = ["//visibility:public"],
    deps = [":java-langtools-location-lib"],
)

# Utility to export javacopts to the java world from the default javabuilder (a
# java_binary deploy jar).
java_binary(
    name = "JavaBuilderConfigGenerator",
    srcs = ["JavaBuilderConfigGenerator.java"],
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
    outs = ["JavaBuilderJavacOpts.java"],
    cmd = "$(location :JavaBuilderConfigGenerator) $< > $@",
    tools = [":JavaBuilderConfigGenerator"],
)

# Provides java-level access to the default javacopts in the current
# JavaBuilder release as defined by bazel's --javabuilder_top flag
java_library(
    name = "javabuilder-javacopts-lib",
    srcs = [":javabuilder-javacopts"],
)

java_library(
    name = "JavaBuilderConfig",
    srcs = ["JavaBuilderConfig.java"],
    data = ["//tools/defaults:javabuilder"],
    deps = [":javabuilder-javacopts-lib"],
)

# Provides programmatic access to a bazel compatible javac.
# Use this instead of ToolProvider.getSystemJavaCompiler().
java_library(
    name = "BazelJavaCompiler",
    srcs = ["BazelJavaCompiler.java"],
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
    srcs = ["BazelJavac.java"],
    deps = [":BazelJavaCompiler"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/java_tools/buildjar:__pkg__"],
)
