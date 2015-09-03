# TODO(dmarting): to make clearer, instead of doing that, we should
# move every target of the BUILD file into there java package.
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
