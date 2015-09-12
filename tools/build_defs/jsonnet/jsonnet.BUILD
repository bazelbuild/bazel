package(default_visibility = ["//visibility:public"])

filegroup(
    name = "std",
    srcs = ["std.jsonnet"],
)

genrule(
    name = "gen-std-jsonnet-h",
    srcs = ["std.jsonnet"],
    outs = ["std.jsonnet.h"],
    cmd = "((od -v -Anone -t u1 $< | tr \" \" \"\n\" | grep -v \"^$$\" " +
          "| tr \"\n\" \",\" ) && echo \"0\") > $@; " +
          "echo >> $@",
)

cc_library(
    name = "jsonnet-common",
    srcs = [
        "lexer.cpp",
        "parser.cpp",
        "static_analysis.cpp",
        "vm.cpp",
        "std.jsonnet.h",
    ],
    hdrs = [
        "lexer.h",
        "parser.h",
        "static_analysis.h",
        "static_error.h",
        "vm.h",
    ],
    linkopts = ["-lm"],
    includes = ["."],
)

cc_library(
    name = "libjsonnet",
    srcs = ["libjsonnet.cpp"],
    hdrs = ["libjsonnet.h"],
    deps = [":jsonnet-common"],
    includes = ["."],
)

cc_binary(
    name = "jsonnet",
    srcs = ["jsonnet.cpp"],
    deps = [":libjsonnet"],
    includes = ["."],
)

cc_binary(
    name = "libjsonnet_test_snippet",
    srcs = ["libjsonnet_test_snippet.c"],
    deps = [":libjsonnet"],
    includes = ["."],
)

cc_binary(
    name = "libjsonnet_test_file",
    srcs = ["libjsonnet_test_file.c"],
    deps = [":libjsonnet"],
    includes = ["."],
)

filegroup(
    name = "object_jsonnet",
    srcs = ["test_suite/object.jsonnet"],
)

sh_test(
    name = "libjsonnet_test",
    srcs = ["libjsonnet_test.sh"],
    data = [
        ":jsonnet",
        ":libjsonnet_test_snippet",
        ":libjsonnet_test_file",
        ":object_jsonnet",
    ],
)
