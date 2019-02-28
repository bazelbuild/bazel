load(
    "//tools/build_rules:test_rules.bzl",
    "rule_test",
    "file_test",
)

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/test/starlark/skylint:srcs"],
)

[
    py_test(
        name = "starlark_test_" + test_file.replace(".", "_"),
        srcs = [
            "starlark_test.py",
            "testenv.py",
        ],
        args = [test_file],
        data = [
            "//src/main/java/com/google/devtools/starlark:Starlark",
            test_file,
        ],
        main = "starlark_test.py",
    )
    for test_file in glob(["testdata/*"])
]

## Rest of the file should be moved somewhere else (under bazel/tools/).

genrule(
    name = "tone_below",
    outs = [
        "g1",
        "g2",
    ],
    cmd = "echo 48.9994 > $(location :g1) ; echo 97.9989 > $(location :g2)",
)

rule_test(
    name = "assert_tone_below_1",
    generates = [
        "g1",
        "g2",
    ],
    rule = "tone_below",
)

file_test(
    name = "question_content",
    content = ("If you could ask a unique question to a computer during a Turing test, " +
               "what would you ask?\n"),
    file = "question.text",
)

file_test(
    name = "question_regexp",
    file = "question.text",
    regexp = "[eu]n[uchs questi]\\+on",
)

file_test(
    name = "question_regexp_1",
    file = "question.text",
    matches = 1,
    regexp = "what would you ask",
)
