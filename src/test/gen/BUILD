# Build rules to test genrules under bazel.
# TODO(bazel-team): Convert to unit test, or remove when the temporary Docker
# override flags are removed.

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

# A rule that can help check if Docker container is enabled.
genrule(
    name = "hostname",
    outs = ["hostname.txt"],
    cmd = "hostname > $@",
)

# A rule to help check if the right uid is set.
genrule(
    name = "uid",
    outs = ["uid.txt"],
    cmd = "id -u > $@; (whoami >> $@ || echo 'No whoami!')",
)

# A genrule to check if output/stdout/strerr are piped correctly.
genrule(
    name = "stdout-stderr",
    outs = ["output.txt"],
    cmd = "echo 'To file.' > $@; echo 'to stdout.'; (>&2 echo 'to stderr.');",
)

# A use case that fails if it's built in a Docker container running under root.
genrule(
    name = "mine",
    outs = ["mine.txt"],
    cmd = "echo 'this is mine!' > $@; chmod 700 $@;",
)

# A use case that fails if it's built in a Docker container running under root.
genrule(
    name = "vars",
    outs = ["vars.txt"],
    cmd = "set > $@",
)
