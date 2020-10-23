filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//third_party/rules_python/python:srcs"],
    visibility = ["//third_party:__pkg__"],
)

# The file is called rules_python.WORKSPACE, but renamed to WORKSPACE
# via new_local_repository, so this works out fine.
exports_files(
    ["WORKSPACE"],
    visibility = ["//visibility:public"],
)
