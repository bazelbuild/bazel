package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "shell",
    srcs = glob([
        "shell/*.java",
    ]),
    deps = ["//third_party:guava"],
)

java_library(
    name = "cmdline",
    srcs = glob([
        "cmdline/*.java",
    ]),
    deps = [
        ":base-util",
        ":common",
        ":concurrent",
        ":syntax",
        ":util",
        ":vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "collect",
    srcs = glob([
        "collect/**/*.java",
    ]),
    deps = [
        ":common",
        ":shell",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "common",
    srcs = [
        "Constants.java",
    ],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "concurrent",
    srcs = glob([
        "concurrent/*.java",
    ]),
    deps = [
        ":common",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "events",
    srcs = glob([
        "events/*.java",
    ]),
    deps = [
        ":common",
        ":concurrent",
        ":io",
        ":vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "syntax",
    srcs = glob(["syntax/**/*.java"]),
    deps = [
        ":base-util",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_proto",
        "//third_party:asm",
        "//third_party:asm-util",
        "//third_party:auto_value",
        "//third_party:bytebuddy",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "packages",
    srcs = glob([
        "packages/*.java",
        "pkgcache/*.java",
    ]),
    exports = [
        ":cmdline",
        ":syntax",
    ],
    deps = [
        ":cmdline",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":syntax",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "unix",
    srcs = glob([
        "unix/*.java",
    ]) + [
        "UnixJniLoader.java",
    ],
    deps = [
        ":common",
        ":shell",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "vfs",
    srcs = glob([
        "vfs/**/*.java",
        "profiler/*.java",
    ]),
    deps = [
        ":base-util",
        ":clock",
        ":common",
        ":concurrent",
        ":os_util",
        ":shell",
        ":unix",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "profiler-output",
    srcs = glob([
        "profiler/chart/*.java",
        "profiler/output/*.java",
        "profiler/statistics/*.java",
    ]),
    deps = [
        ":collect",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# Clock classes.
java_library(
    name = "clock",
    srcs = [
        "util/BlazeClock.java",
        "util/Clock.java",
        "util/JavaClock.java",
    ],
    deps = [
        ":concurrent",
    ],
)

# String canonicalizer.
java_library(
    name = "base-util",
    srcs = [
        "util/StringCanonicalizer.java",
        "util/StringTrie.java",
        "util/VarInt.java",
    ],
    deps = [
        ":common",
        "//third_party:guava",
    ],
)

# I/O utilities.
java_library(
    name = "io",
    srcs = glob(["util/io/*.java"]),
    deps = [
        ":clock",
        ":common",
        ":concurrent",
        ":vfs",
        "//third_party:guava",
    ],
)

# General utilities.
java_library(
    name = "os_util",
    srcs = ["util/OS.java"],
)

java_library(
    name = "util",
    srcs = glob(
        ["util/*.java"],
        exclude = [
            "util/BlazeClock.java",
            "util/Clock.java",
            "util/JavaClock.java",
            "util/OS.java",
            "util/StringCanonicalizer.java",
            "util/StringTrie.java",
            "util/VarInt.java",
        ],
    ),
    deps = [
        ":base-util",
        ":clock",
        ":collect",
        ":common",
        ":concurrent",
        ":os_util",
        ":shell",
        ":unix",
        ":vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "analysis-exec-rules-skyframe",
    srcs = glob(
        [
            "analysis/**/*.java",
            "exec/*.java",
            "ideinfo/**/*.java",
            "rules/**/*.java",
            "skyframe/*.java",
        ],
        exclude = [
            "rules/java/JavaToolchainDataParser.java",
        ],
    ) + [
        "runtime/BlazeServerStartupOptions.java",
    ],
    deps = [
        ":base-util",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":graph",
        ":io",
        ":os_util",
        ":packages",
        ":shell",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:android_studio_ide_info_proto",
        "//src/main/protobuf:bundlemerge_proto",
        "//src/main/protobuf:crosstool_config_proto",
        "//src/main/protobuf:extra_actions_base_proto",
        "//src/main/protobuf:test_status_proto",
        "//src/main/protobuf:xcodegen_proto",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/common",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/util",
        "//third_party:guava",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

java_library(
    name = "graph",
    srcs = glob([
        "graph/*.java",
    ]),
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "query2",
    srcs = glob([
        "query2/**/*.java",
    ]),
    deps = [
        ":analysis-exec-rules-skyframe",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":graph",
        ":packages",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "buildtool-runtime",
    srcs = glob([
        "buildtool/**/*.java",
        "runtime/**/*.java",
    ]),
    deps = [
        ":analysis-exec-rules-skyframe",
        ":clock",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":io",
        ":os_util",
        ":packages",
        ":profiler-output",
        ":query2",
        ":server",
        ":shell",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/build/docgen",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_proto",
        "//src/main/protobuf:invocation_policy_proto",
        "//src/main/protobuf:test_status_proto",
        "//third_party:guava",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

java_library(
    name = "runtime",
    srcs = glob([
        "runtime/**/*.java",
        "buildtool/**/*.java",
        "server/**/*.java",
    ]),
    deps = [
        ":analysis-exec-rules-skyframe",
        ":clock",
        ":cmdline",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":io",
        ":os_util",
        ":packages",
        ":profiler-output",
        ":query2",
        ":shell",
        ":unix",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/build/docgen",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_proto",
        "//src/main/protobuf:invocation_policy_proto",
        "//src/main/protobuf:test_status_proto",
        "//third_party:guava",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

java_library(
    name = "server",
    srcs = glob([
        "server/**/*.java",
    ]),
    deps = [
        ":clock",
        ":io",
        ":unix",
        ":util",
        ":vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "bazel-core",
    srcs = glob(
        [
            "bazel/**/*.java",
        ],
        exclude = [
            "bazel/repository/MavenConnector.java",
        ],
    ),
    resources = glob([
        "**/*.txt",
        "**/*.html",
        "**/*.css",
        "**/*.js",
        "**/*.WORKSPACE",
        "**/*.vm",
    ]) + [
        "//third_party:d3-js",
        "//third_party:jquery",
    ],
    visibility = [
        "//src:__subpackages__",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace:__pkg__",
    ],
    runtime_deps = [
        "//third_party:aether",
        "//third_party:apache_commons_codec",
        "//third_party:apache_commons_logging",
        "//third_party:apache_httpclient",
        "//third_party:apache_httpcore",
        "//third_party:jsch",
        "//third_party:maven",
        "//third_party:maven_model",
        "//third_party:plexus_interpolation",
        "//third_party:plexus_utils",
    ],
    deps = [
        ":analysis-exec-rules-skyframe",
        ":buildtool-runtime",
        ":clock",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":graph",
        ":io",
        ":maven-connector",
        ":os_util",
        ":packages",
        ":query2",
        ":runtime",
        ":shell",
        ":unix",
        ":util",
        ":vfs",
        "//src/java_tools/singlejar:zip",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/bazel/dash",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/standalone",
        "//src/main/java/com/google/devtools/build/lib/worker",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_proto",
        "//src/main/protobuf:crosstool_config_proto",
        "//src/main/protobuf:extra_actions_base_proto",
        "//src/main/protobuf:test_status_proto",
        "//src/main/protobuf:worker_protocol_proto",
        "//third_party:aether",
        "//third_party:apache_commons_compress",
        "//third_party:apache_commons_pool2",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jgit",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:jsr330_inject",
        "//third_party:maven",
        "//third_party:maven_model",
        "//third_party:plexus_component_annotations",
        "//third_party:protobuf",
        "//third_party:slf4j",
    ],
)

java_library(
    name = "maven-connector",
    srcs = ["bazel/repository/MavenConnector.java"],
    deps = [
        "//third_party:aether",
        "//third_party:jsr330_inject",
        "//third_party:maven_model",
    ],
)

java_library(
    name = "java-toolchain-parser",
    testonly = 1,
    srcs = glob([
        # JavaToolchain parser
        "rules/java/JavaToolchainData.java",
        "rules/java/JavaToolchainDataParser.java",
        # Concurrent stuff
        "concurrent/*.java",
    ]),
    visibility = [
        "//src/java_tools/buildjar:__pkg__",
    ],
    deps = [
        ":common",
        "//src/main/protobuf:build_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

java_binary(
    name = "bazel-main",
    main_class = "com.google.devtools.build.lib.bazel.BazelMain",
    visibility = ["//src:__pkg__"],
    runtime_deps = [
        ":bazel-core",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/common/options:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/dash:srcs",
        "//src/main/java/com/google/devtools/build/lib/sandbox:srcs",
        "//src/main/java/com/google/devtools/build/skyframe:srcs",
        "//src/main/java/com/google/devtools/build/lib/standalone:srcs",
        "//src/main/java/com/google/devtools/build/lib/worker:srcs",
        "//src/main/java/com/google/devtools/build/docgen:srcs",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "docs_embedded_in_sources",
    srcs = glob(["**/*.java"]),
)

genrule(
    name = "gen_buildencyclopedia",
    srcs = [":docs_embedded_in_sources"],
    outs = ["build-encyclopedia.zip"],
    cmd = (
        "mkdir -p $(@D)/be && " +
        "$(location //src/main/java/com/google/devtools/build/docgen:docgen_bin)" +
        " $$PWD/src/main/java/com/google/devtools/build/lib" +
        " $(@D)/be &&" +
        "zip -qj $@ $(@D)/be/*"
    ),
    tools = [
        "//src/main/java/com/google/devtools/build/docgen:docgen_bin",
        "//src/main/java/com/google/devtools/build/docgen:docgen_javalib",
    ],
)

genrule(
    name = "gen_skylarklibrary",
    outs = ["skylark-library.zip"],
    cmd = "mkdir -p $(@D)/skylark-lib &&" +
          "$(location //src/main/java/com/google/devtools/build/docgen:skydoc_bin) $(@D)/skylark-lib &&" +
          "zip -qj $@ $(@D)/skylark-lib/*",
    tools = ["//src/main/java/com/google/devtools/build/docgen:skydoc_bin"],
)

#
# Bootstrapping SingleJar using Skylark rules
#
load("/tools/build_rules/java_rules_skylark", "bootstrap_java_library")

bootstrap_java_library(
    name = "shell-skylark",
    srcs = glob(["shell/*.java"]),
    jars = ["//third_party:guava"],
)
