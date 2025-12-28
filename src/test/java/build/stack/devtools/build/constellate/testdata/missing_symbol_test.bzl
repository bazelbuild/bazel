# Test file that loads a missing/renamed symbol

load("@rules_cc//cc:find_cc_toolchain.bzl", "use_cc_toolchain")

# This should be gracefully ignored since use_cc_toolchain was renamed/removed
def test_function():
    """Test function."""
    pass
