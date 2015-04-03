package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "libSingleJar",
    srcs = glob(["java/**/singlejar/**/*.java"]),
    deps = [
        ":zip",
        "//src/main/java:shell",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_binary(
    name = "SingleJar",
    main_class = "com.google.devtools.build.singlejar.SingleJar",
    runtime_deps = [":libSingleJar"],
)

java_test(
    name = "tests",
    srcs = glob(["javatests/**/singlejar/**/*.java"]),
    args = ["com.google.devtools.build.singlejar.SingleJarTests"],
    deps = [
        ":libSingleJar",
        ":zip",
        "//src/main/java:shell",
        "//src/test/java:testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "zip",
    srcs = glob(["java/**/zip/**/*.java"]),
    deps = [
        "//third_party:jsr305",
    ],
)

java_test(
    name = "zipTests",
    srcs = glob(["javatests/**/zip/**/*.java"]),
    args = ["com.google.devtools.build.zip.ZipTests"],
    deps = [
        ":zip",
        "//src/test/java:testutil",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
