load("@rules_python//python:defs.bzl", "py_binary", "py_library")

BUILD_SCM_REV_CMD = "$$(grep BUILD_SCM_REVISION bazel-out/volatile-status.txt | sed 's/^BUILD_SCM_REVISION //')"

# Requires --config=docs
genrule(
    name = "gen_release_docs",
    srcs = [
        ":new_toc.yaml",
        ":new_buttons.html",
        "//site/en:docs",
        "//src/main/java/com/google/devtools/build/lib:reference-docs.zip",
    ],
    outs = [
        "release_docs.zip",
    ],
    cmd = "$(location :create_release_docs)" +
          " --version=" + BUILD_SCM_REV_CMD +
          " --toc_path=$(location :new_toc.yaml)" +
          " --buttons_path=$(location :new_buttons.html)" +
          " --narrative_docs_path=$(location //site/en:docs)" +
          " --reference_docs_path=$(location //src/main/java/com/google/devtools/build/lib:reference-docs.zip)" +
          " --output_path=$(OUTS)",
    stamp = 1,
    tools = [
        ":create_release_docs",
    ],
)

# Requires --config=docs
genrule(
    name = "gen_new_toc",
    srcs = [
        "//site/en:versions/_toc.yaml",
        "//site/en:_buttons.html",
    ],
    outs = [
        "new_toc.yaml",
        "new_buttons.html",
    ],
    cmd = "$(location //src/main/java/com/google/devtools/build/docgen/release:toc_updater)" +
          " -i $(location //site/en:versions/_toc.yaml)" +
          " -o $(location new_toc.yaml)" +
          " -v " + BUILD_SCM_REV_CMD +
          " --version_indicator_input=$(location //site/en:_buttons.html)" +
          " --version_indicator_output=$(location new_buttons.html)",
    stamp = 1,
    tools = [
        "//src/main/java/com/google/devtools/build/docgen/release:toc_updater",
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
