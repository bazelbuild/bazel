package(default_visibility = ["//visibility:public"])

genquery(
    name = "genquery",
    expression = "deps(//tools/defaults:jdk)",
    scope = ["//tools/defaults:jdk"],
)

genrule(
    name = "genrule",
    srcs = [":genquery"],
    outs = ["genrule.txt"],
    cmd = "cat $(SRCS) > $@",
)

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "genrule.txt",
    ],
)
