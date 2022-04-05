load("//tools/python:private/defs.bzl", "py_binary", "py_library")

BUILD_SCM_REV_CMD = "$$(grep BUILD_SCM_REVISION bazel-out/volatile-status.txt | sed 's/^BUILD_SCM_REVISION //')"

# Requires --config=docs
genrule(
    name = "gen_release_docs",
    srcs = [
        ":new_toc.yaml",
        "//site/en:docs",
        "//src/main/java/com/google/devtools/build/lib:reference-docs.zip",
    ],
    cmd = "$(location :create_release_docs)" +
          " --version=" + BUILD_SCM_REV_CMD + 
          " --toc_path=$(location :new_toc.yaml)" +
          " --narrative_docs_path=$(location //site/en:docs)" +
          " --reference_docs_path=$(location //src/main/java/com/google/devtools/build/lib:reference-docs.zip)" +
          " --output_path=$(OUTS)",
    outs = [
      "release_docs.zip",
    ],
    tools = [
        ":create_release_docs",
    ],
    stamp = 1,
)

# Requires --config=docs
genrule(
    name = "gen_new_toc",
    srcs = [
        "//site/en:versions/_toc.yaml",
    ],
    cmd = "$(location //src/main/java/com/google/devtools/build/docgen:toc_updater)" +
          " -i $(location //site/en:versions/_toc.yaml)" +
          " -o $(OUTS)" +
          " -v " + BUILD_SCM_REV_CMD,
    outs = ["new_toc.yaml"],
    tools = [
        "//src/main/java/com/google/devtools/build/docgen:toc_updater",
    ],
    stamp = 1,
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
    name = "testdata",
    srcs = glob(["testdata/**"]),
    visibility = [
        ":__pkg__",
    ],
)