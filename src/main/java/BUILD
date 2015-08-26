package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "shell",
    srcs = glob([
        "com/google/devtools/build/lib/shell/*.java",
    ]),
    deps = ["//third_party:guava"],
)

java_library(
    name = "actions",
    srcs = glob([
        "com/google/devtools/build/lib/actions/**/*.java",
    ]),
    deps = [
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":options",
        ":packages",
        ":shell",
        ":skyframe-base",
        ":vfs",
        "//src/main/protobuf:proto_extra_actions_base",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

java_library(
    name = "cmdline",
    srcs = glob([
        "com/google/devtools/build/lib/cmdline/*.java",
    ]),
    deps = [
        ":common",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "collect",
    srcs = glob([
        "com/google/devtools/build/lib/collect/**/*.java",
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
        "com/google/devtools/build/lib/Constants.java",
    ],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "concurrent",
    srcs = glob([
        "com/google/devtools/build/lib/concurrent/*.java",
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
        "com/google/devtools/build/lib/events/*.java",
    ]),
    deps = [
        ":common",
        ":concurrent",
        ":vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "options",
    srcs = glob([
        "com/google/devtools/common/options/*.java",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "packages",
    srcs = glob([
        "com/google/devtools/build/lib/packages/*.java",
        "com/google/devtools/build/lib/pkgcache/*.java",
        "com/google/devtools/build/lib/syntax/*.java",
    ]),
    exports = [":cmdline"],
    deps = [
        ":cmdline",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":options",
        ":vfs",
        "//src/main/protobuf:proto_build",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "unix",
    srcs = glob([
        "com/google/devtools/build/lib/unix/*.java",
    ]) + [
        "com/google/devtools/build/lib/UnixJniLoader.java",
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
        "com/google/devtools/build/lib/vfs/**/*.java",
        "com/google/devtools/build/lib/profiler/**/*.java",
        "com/google/devtools/build/lib/util/**/*.java",
    ]),
    deps = [
        ":collect",
        ":common",
        ":concurrent",
        ":options",
        ":shell",
        ":unix",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "skyframe-base",
    srcs = glob([
        "com/google/devtools/build/skyframe/*.java",
    ]),
    deps = [
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "analysis-exec-rules-skyframe",
    srcs = glob(
        [
            "com/google/devtools/build/lib/analysis/**/*.java",
            "com/google/devtools/build/lib/exec/*.java",
            "com/google/devtools/build/lib/rules/**/*.java",
            "com/google/devtools/build/lib/skyframe/*.java",
        ],
        exclude = [
            "com/google/devtools/build/lib/rules/java/JavaToolchainDataParser.java",
            "com/google/devtools/build/lib/rules/genquery/**",
        ],
    ) + [
        "com/google/devtools/build/lib/runtime/BlazeServerStartupOptions.java",
    ],
    deps = [
        ":actions",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":graph",
        ":options",
        ":packages",
        ":shell",
        ":skyframe-base",
        ":vfs",
        "//src/main/protobuf:proto_bundlemerge",
        "//src/main/protobuf:proto_crosstool_config",
        "//src/main/protobuf:proto_extra_actions_base",
        "//src/main/protobuf:proto_test_status",
        "//src/main/protobuf:proto_xcodegen",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/common",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/util",
        "//third_party:guava",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

java_library(
    name = "genquery",
    srcs = glob([
        "com/google/devtools/build/lib/rules/genquery/*.java",
    ]),
    deps = [
        ":actions",
        ":analysis-exec-rules-skyframe",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":graph",
        ":options",
        ":packages",
        ":query2",
        ":shell",
        ":skyframe-base",
        ":vfs",
        "//src/main/protobuf:proto_crosstool_config",
        "//src/main/protobuf:proto_extra_actions_base",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

java_library(
    name = "graph",
    srcs = glob([
        "com/google/devtools/build/lib/graph/*.java",
    ]),
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "query2",
    srcs = glob([
        "com/google/devtools/build/lib/query2/**/*.java",
    ]),
    deps = [
        ":analysis-exec-rules-skyframe",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":graph",
        ":options",
        ":packages",
        ":skyframe-base",
        ":vfs",
        "//src/main/protobuf:proto_build",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "buildtool-runtime",
    srcs = glob([
        "com/google/devtools/build/lib/buildtool/**/*.java",
        "com/google/devtools/build/lib/runtime/**/*.java",
    ]),
    deps = [
        ":actions",
        ":analysis-exec-rules-skyframe",
        ":collect",
        ":common",
        ":concurrent",
        ":docgen",
        ":events",
        ":options",
        ":packages",
        ":query2",
        ":server",
        ":shell",
        ":skyframe-base",
        ":vfs",
        "//src/main/protobuf:proto_build",
        "//src/main/protobuf:proto_test_status",
        "//third_party:guava",
        "//third_party:joda_time",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "docgen",
    srcs = glob([
        "com/google/devtools/build/docgen/**/*.java",
    ]),
    deps = [
        ":analysis-exec-rules-skyframe",
        ":common",
        ":concurrent",
        ":packages",
        ":vfs",
        "//third_party:apache_velocity",
        "//third_party:guava",
    ],
)

java_library(
    name = "runtime",
    srcs = glob([
        "com/google/devtools/build/lib/runtime/**/*.java",
        "com/google/devtools/build/lib/buildtool/**/*.java",
        "com/google/devtools/build/lib/server/**/*.java",
    ]),
    deps = [
        ":actions",
        ":analysis-exec-rules-skyframe",
        ":cmdline",
        ":collect",
        ":common",
        ":concurrent",
        ":docgen",
        ":events",
        ":options",
        ":packages",
        ":query2",
        ":shell",
        ":skyframe-base",
        ":unix",
        ":vfs",
        "//src/main/protobuf:proto_build",
        "//src/main/protobuf:proto_test_status",
        "//third_party:guava",
        "//third_party:joda_time",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "server",
    srcs = glob([
        "com/google/devtools/build/lib/server/**/*.java",
    ]),
    deps = [
        ":unix",
        ":vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "webstatusserver",
    srcs = glob([
        "com/google/devtools/build/lib/webstatusserver/**/*.java",
    ]),
    deps = [
        ":analysis-exec-rules-skyframe",
        ":buildtool-runtime",
        ":events",
        ":options",
        ":packages",
        ":vfs",
        "//src/main/protobuf:proto_test_status",
        "//third_party:gson",
        "//third_party:guava",
    ],
)

java_library(
    name = "bazel-core",
    srcs = glob(
        [
            "com/google/devtools/build/lib/bazel/**/*.java",
        ],
        exclude = [
            "com/google/devtools/build/lib/bazel/repository/MavenConnector.java",
        ],
    ),
    resources = glob([
        "com/google/devtools/build/lib/**/*.txt",
        "com/google/devtools/build/lib/**/*.html",
        "com/google/devtools/build/lib/**/*.css",
        "com/google/devtools/build/lib/**/*.js",
        "com/google/devtools/build/lib/**/*.WORKSPACE",
        "com/google/devtools/build/lib/**/*.vm",
    ]) + [
        "//third_party:d3-js",
        "//third_party:jquery",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/workspace:__pkg__",
        "//src/test/java:__subpackages__",
    ],
    runtime_deps = [
        "//third_party:aether",
        "//third_party:apache_commons_logging",
        "//third_party:apache_httpclient",
        "//third_party:apache_httpcore",
        "//third_party:maven_model",
        "//third_party:plexus_interpolation",
        "//third_party:plexus_utils",
    ],
    deps = [
        ":actions",
        ":analysis-exec-rules-skyframe",
        ":buildtool-runtime",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":genquery",
        ":graph",
        ":maven-connector",
        ":options",
        ":packages",
        ":query2",
        ":runtime",
        ":shell",
        ":skyframe-base",
        ":unix",
        ":vfs",
        ":webstatusserver",
        "//src/java_tools/singlejar:zip",
        "//src/main/java/com/google/devtools/build/lib/bazel/dash",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/standalone",
        "//src/main/java/com/google/devtools/build/lib/worker",
        "//src/main/protobuf:proto_build",
        "//src/main/protobuf:proto_crosstool_config",
        "//src/main/protobuf:proto_extra_actions_base",
        "//src/main/protobuf:proto_test_status",
        "//src/main/protobuf:proto_worker_protocol",
        "//third_party:aether",
        "//third_party:apache_commons_compress",
        "//third_party:apache_commons_pool2",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jgit",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:jsr330_inject",
        "//third_party:maven_model",
        "//third_party:protobuf",
        "//third_party:slf4j",
    ],
)

java_library(
    name = "maven-connector",
    srcs = ["com/google/devtools/build/lib/bazel/repository/MavenConnector.java"],
    deps = [
        "//third_party:aether",
        "//third_party:maven_model",
    ],
)

java_library(
    name = "java-toolchain-parser",
    testonly = 1,
    srcs = glob([
        # JavaToolchain parser
        "com/google/devtools/build/lib/rules/java/JavaToolchainData.java",
        "com/google/devtools/build/lib/rules/java/JavaToolchainDataParser.java",
        # Concurrent stuff
        "com/google/devtools/build/lib/concurrent/*.java",
    ]),
    visibility = [
        "//src/java_tools/buildjar:__pkg__",
    ],
    deps = [
        ":common",
        "//src/main/protobuf:proto_build",
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
        "//src/main/java/com/google/devtools/build/lib/bazel/dash:srcs",
        "//src/main/java/com/google/devtools/build/lib/sandbox:srcs",
        "//src/main/java/com/google/devtools/build/lib/standalone:srcs",
        "//src/main/java/com/google/devtools/build/lib/worker:srcs",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "be-template_files",
    srcs = glob([
        "com/google/devtools/build/docgen/templates/**/*.html",
        "com/google/devtools/build/docgen/templates/**/*.vm",
    ]),
)

java_library(
    name = "docgen_javalib",
    srcs = glob(
        ["com/google/devtools/build/docgen/**/*.java"],
        exclude = [
            "com/google/devtools/build/docgen/BuildEncyclopediaGenerator.java",
            "com/google/devtools/build/docgen/SkylarkDocumentationGenerator.java",
        ],
    ),
    resources = [":be-template_files"],
    exports = [
        ":bazel-core",
    ],
    deps = [
        ":analysis-exec-rules-skyframe",
        ":bazel-core",
        ":common",
        ":concurrent",
        ":packages",
        ":vfs",
        "//third_party:apache_velocity",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "docs_embedded_in_sources",
    srcs = glob(["com/google/devtools/build/lib/**/*.java"]),
)

java_binary(
    name = "docgen_bin",
    srcs = ["com/google/devtools/build/docgen/BuildEncyclopediaGenerator.java"],
    main_class = "com.google.devtools.build.docgen.BuildEncyclopediaGenerator",
    deps = [
        ":analysis-exec-rules-skyframe",
        ":common",
        ":docgen_javalib",
    ],
)

java_binary(
    name = "skydoc_bin",
    srcs = ["com/google/devtools/build/docgen/SkylarkDocumentationGenerator.java"],
    main_class = "com.google.devtools.build.docgen.SkylarkDocumentationGenerator",
    deps = [":docgen_javalib"],
)

genrule(
    name = "gen_buildencyclopedia",
    srcs = [":docs_embedded_in_sources"],
    outs = ["build-encyclopedia.html"],
    cmd = ("$(location docgen_bin)" +
           " $$PWD/src/main/java/com/google/devtools/build/lib" +
           " $$PWD" +
           " && cp $$PWD/build-encyclopedia.html $@"),
    tools = [
        ":docgen_bin",
        ":docgen_javalib",
    ],
)

genrule(
    name = "gen_skylarklibrary",
    outs = ["skylark-library.zip"],
    cmd = "mkdir -p $(@D)/skylark-lib &&" +
          "$(location :skydoc_bin) $(@D)/skylark-lib &&" +
          "zip -qj $@ $(@D)/skylark-lib/*",
    tools = [":skydoc_bin"],
)

#
# Bootstrapping SingleJar using Skylark rules
#
load("/tools/build_rules/java_rules_skylark", "bootstrap_java_library")

bootstrap_java_library(
    name = "shell-skylark",
    srcs = glob(["com/google/devtools/build/lib/shell/*.java"]),
    jars = ["//third_party:guava"],
)
