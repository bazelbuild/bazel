# Description:
#   Documentation generator for Bazel
package(
    default_visibility = ["//src:__subpackages__"],
)

java_library(
    name = "docgen_javalib",
    srcs = glob(["**/*.java"]),
    resources = [":template_files"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:android-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:classpath-util",
        "//src/main/java/com/google/devtools/build/lib:java-compilation",
        "//src/main/java/com/google/devtools/build/lib:java-rules",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/swift",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/objc",
        "//src/main/java/com/google/devtools/build/lib/rules/platform",
        "//src/main/java/com/google/devtools/common/options",
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
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_binary(
    name = "skydoc_bin",
    srcs = ["SkylarkDocumentationGenerator.java"],
    main_class = "com.google.devtools.build.docgen.SkylarkDocumentationGenerator",
    runtime_deps = ["//src/main/java/com/google/devtools/build/lib:bazel-repository"],
    deps = [":docgen_javalib"],
)

filegroup(
    name = "template_files",
    srcs = glob([
        "templates/**/*.html",
        "templates/**/*.vm",
    ]),
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
