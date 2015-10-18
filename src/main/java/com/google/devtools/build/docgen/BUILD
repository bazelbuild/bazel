package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "docgen",
    srcs = glob([
        "**/*.java",
    ]),
    deps = [
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:common",
        "//src/main/java:concurrent",
        "//src/main/java:packages",
        "//src/main/java:util",
        "//third_party:apache_velocity",
        "//third_party:guava",
    ],
)

java_library(
    name = "docgen_javalib",
    srcs = glob(
        ["**/*.java"],
        exclude = [
            "BuildEncyclopediaGenerator.java",
            "SkylarkDocumentationGenerator.java",
        ],
    ),
    resources = [":be-template_files"],
    exports = [
        "//src/main/java:bazel-core",
    ],
    deps = [
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:common",
        "//src/main/java:concurrent",
        "//src/main/java:packages",
        "//src/main/java:util",
        "//third_party:apache_velocity",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_binary(
    name = "docgen_bin",
    srcs = ["BuildEncyclopediaGenerator.java"],
    main_class = "com.google.devtools.build.docgen.BuildEncyclopediaGenerator",
    deps = [
        ":docgen_javalib",
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:common",
    ],
)

filegroup(
    name = "be-template_files",
    srcs = glob([
        "templates/**/*.html",
        "templates/**/*.vm",
    ]),
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_binary(
    name = "skydoc_bin",
    srcs = ["SkylarkDocumentationGenerator.java"],
    main_class = "com.google.devtools.build.docgen.SkylarkDocumentationGenerator",
    deps = [":docgen_javalib"],
)
