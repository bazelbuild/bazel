package(default_visibility = ["//visibility:public"])

load(
    "//tools/build_defs/jsonnet:jsonnet.bzl",
    "jsonnet_library",
    "jsonnet_to_json",
    "jsonnet_to_json_test",
)

jsonnet_library(
    name = "workflow",
    srcs = ["workflow.jsonnet"],
)

jsonnet_to_json(
    name = "wordcount",
    src = "wordcount.jsonnet",
    outs = ["wordcount.json"],
    deps = [":workflow"],
)

jsonnet_to_json_test(
    name = "wordcount_test",
    src = "wordcount.jsonnet",
    golden = "wordcount_golden.json",
    deps = [":workflow"],
)

jsonnet_to_json(
    name = "intersection",
    src = "intersection.jsonnet",
    outs = ["intersection.json"],
    deps = [":workflow"],
)

jsonnet_to_json_test(
    name = "intersection_test",
    src = "intersection.jsonnet",
    golden = "intersection_golden.json",
    deps = [":workflow"],
)

jsonnet_library(
    name = "shell-workflows-lib",
    srcs = [
        "intersection.jsonnet",
        "wordcount.jsonnet",
    ],
    deps = [":workflow"],
)

jsonnet_to_json(
    name = "shell-workflows",
    src = "shell-workflows.jsonnet",
    outs = [
        "intersection-workflow.json",
        "wordcount-workflow.json",
    ],
    deps = [":shell-workflows-lib"],
)

jsonnet_to_json_test(
    name = "invalid_test",
    src = "invalid.jsonnet",
    error = 1,
    golden = "invalid.out",
)
