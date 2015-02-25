package(default_visibility = ["//src:__pkg__"])

java_library(
    name = "libSingleJar",
    srcs = glob(["java/**/*.java"]),
    deps = [
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
    srcs = glob(["javatests/**/*.java"]),
    args = ["com.google.devtools.build.singlejar.SingleJarTests"],
    deps = [
        ":libSingleJar",
        "//src/main/java:shell",
        "//src/test/java:testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
