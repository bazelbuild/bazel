"""Test file with a failing load statement to test best-effort extraction."""

# This load should fail (file doesn't exist)
# load(":nonexistent_file.bzl", "nonexistent_function")

# But we should still be able to extract these entities defined in this file
def local_function():
    """A function defined locally.

    Returns:
        A string value
    """
    return "local"

LocalInfo = provider(
    doc = "A provider defined locally despite load failure.",
    fields = ["local_value"],
)

def _local_rule_impl(ctx):
    return [LocalInfo(local_value = "local")]

local_rule = rule(
    implementation = _local_rule_impl,
    doc = "A rule defined locally despite load failure.",
    attrs = {},
)
