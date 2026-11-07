load("@rules_shell//shell:sh_test.bzl", "sh_test")

# Harmless marker test for Google VRP issue 582019927.
sh_test(
    name = "marker_test",
    srcs = ["marker_test.sh"],
    tags = [
        "requires-network",
        "no-remote-cache",
        "no-remote-exec",
    ],
)
