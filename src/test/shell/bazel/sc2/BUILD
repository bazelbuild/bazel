load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
    name = "sc2_probe_test",
    size = "small",
    srcs = ["sc2_probe_test.sh"],
    tags = [
        "requires-network",
    ],
    timeout = "moderate",
)
