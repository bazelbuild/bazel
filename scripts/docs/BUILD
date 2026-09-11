load("@bazel_pip_dev_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

BUILD_SCM_REV_CMD = "\\\"$$(grep BUILD_SCM_REVISION bazel-out/volatile-status.txt | sed 's/^BUILD_SCM_REVISION //')\\\""

# Requires --config=docs
# TODO: Rename to gen_release_docs once the docs migration is done.
genrule(
    name = "gen_mdx_release_docs",
    srcs = [
        "//docs",
        "//src/main/java/com/google/devtools/build/lib:mdx-reference-docs.zip",
    ],
    outs = [
        "mdx_release_docs.zip",
    ],
    cmd = "$(location :create_release_docs)" +
          " --version=" + BUILD_SCM_REV_CMD +
          " --narrative_docs_path=$(location //docs)" +
          " --reference_docs_path=$(location //src/main/java/com/google/devtools/build/lib:mdx-reference-docs.zip)" +
          " --output_path=$(OUTS)",
    stamp = 1,
    tools = [
        ":create_release_docs",
    ],
)

py_binary(
    name = "create_release_docs",
    srcs = ["create_release_docs.py"],
    deps = [
        ":rewriter",
        "//third_party/py/abseil",
    ],
)

py_library(
    name = "rewriter",
    srcs = ["rewriter.py"],
)

py_test(
    name = "rewriter_test",
    srcs = ["rewriter_test.py"],
    data = [":testdata"],
    deps = [
        ":rewriter",
        "//third_party/py/abseil",
    ],
)

py_binary(
    name = "create_cli_reference",
    srcs = ["create_cli_reference.py"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:__pkg__",
    ],
    deps = [
        "//src/main/protobuf:bazel_flags_py_proto",
    ],
)

py_library(
    name = "docs2mdx_lib",
    srcs = ["docs2mdx.py"],
    deps = [
        "//third_party/py/abseil",
        requirement("markdownify"),
    ],
)

py_binary(
    name = "docs2mdx",
    srcs = ["docs2mdx.py"],
    main = "docs2mdx.py",
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:__pkg__",
    ],
    deps = [
        ":docs2mdx_lib",
    ],
)

py_test(
    name = "docs2mdx_test",
    srcs = ["docs2mdx_test.py"],
    deps = [
        ":docs2mdx_lib",
        "//third_party/py/abseil",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//scripts:__pkg__"],
)

filegroup(
    name = "testdata",
    srcs = glob(["testdata/**"]),
    visibility = [
        ":__pkg__",
    ],
)
