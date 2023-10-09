load(":generate_lock_file.bzl", "generate_lock_file")

# To avoid accessing BCR for most tests, we generate a default MODULE.bazel.lock file
# in an empty workspace that will be used in integration tests setup.
generate_lock_file(
    name = "gen_bzlmod_lock_file",
    testonly = True,
    visibility = ["//src/test:__subpackages__"],
)
