package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

sh_binary(
    name = "zip_manifest_creator",
    srcs = ["zip_manifest_creator.sh"],
    data = ["//third_party/ijar:zipper"],
)

sh_test(
    name = "zip_manifest_creator_test",
    size = "small",
    srcs = ["zip_manifest_creator_test.sh"],
    data = [":zip_manifest_creator"],
)

py_binary(
    name = "embedded_jar_extractor",
    srcs = ["embedded_jar_extractor.py"],
    deps = ["//third_party/py/gflags"],
)

py_test(
    name = "embedded_jar_extractor_test",
    srcs = ["embedded_jar_extractor_test.py"],
    deps = [":embedded_jar_extractor"],
)
