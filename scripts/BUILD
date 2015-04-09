# Scripts for IDE / Environment support for Bazel
package(default_visibility = ["//visibility:private"])

# The master copy of the completion logic are bazel-complete-template.bash
# and bazel-complete-header.bash.
# This is where you should make edits.
genrule(
    name = "bash_completion",
    srcs = ["bazel-complete-template.bash"],
    outs = ["bazel-complete.bash"],
    cmd = "set +x; cat $(SRCS) > $@\n" +
          "touch WORKSPACE err.log\n" +
          "trap \"rm WORKSPACE err.log\" EXIT\n" +
          "$(location //src:bazel) help completion 2>err.log >>$@ || { cat err.log; exit 1; }",
    output_to_bindir = 1,
    tools = ["//src:bazel"],
)

sh_test(
    name = "bash_completion_test",
    size = "small",
    srcs = ["bash_completion_test.sh"],
    data = [
        "bazel-complete.bash",
        "testenv.sh",
        "//src/test/shell:bashunit",
    ],
)
