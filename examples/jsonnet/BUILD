package(default_visibility = ["//visibility:public"])

load("/tools/build_defs/jsonnet/jsonnet", "jsonnet_library", "jsonnet_to_json")

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

jsonnet_to_json(
    name = "intersection",
    src = "intersection.jsonnet",
    outs = ["intersection.json"],
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
