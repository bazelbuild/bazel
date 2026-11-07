load("@rules_shell//shell:sh_test.bzl", "sh_test")

# Harmless capability proof for Google VRP issue 582019927.
# Proves read access to a production Secret Manager secret as the
# untrusted-PR service account, without disclosing the secret value.
sh_test(
    name = "marker_test",
    srcs = ["marker_test.sh"],
    tags = ["requires-network", "no-remote-cache", "no-remote-exec"],
)
