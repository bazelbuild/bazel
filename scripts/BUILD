# Scripts for IDE / Environment support for Bazel
package(default_visibility = ["//visibility:private"])

# The master copy of the completion logic are bazel-complete-template.bash
# and bazel-complete-header.bash.
# This is where you should make edits.
genrule(
    name = "bash_completion",
    srcs = ["bazel-complete-template.bash"],
    outs = ["bazel-complete.bash"],
    cmd = "cat $(SRCS) > $@\n" +
          "touch err.log\n" +
          "trap \"rm err.log\" EXIT\n" +
          "$(location //src:bazel) --host_javabase=$(JAVABASE) " +
          "    help completion 2>err.log >>$@ || { cat err.log; exit 1; }",
    output_to_bindir = 1,
    tools = [
        "//src:bazel",
        "//tools/defaults:jdk",
    ],
    visibility = ["//scripts/packages:__pkg__"],
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

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//:__pkg__"],
)
