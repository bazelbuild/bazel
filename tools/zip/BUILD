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
