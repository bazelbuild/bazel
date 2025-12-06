# Test using provider as dictionary key

MyProvider = provider(
    fields = ["value"],
)

# This should work - providers should be hashable
config_dict = {
    MyProvider: "some config",
}

def test_function():
    """Test that providers can be used as dict keys."""
    return config_dict
