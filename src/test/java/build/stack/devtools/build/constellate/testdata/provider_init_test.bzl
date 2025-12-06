# Test provider() with init parameter returning a 2-element tuple

def _create_proto_info(**kwargs):
    """Custom init function for ProtoInfo provider."""
    return kwargs

# This should return a tuple (Provider, raw_constructor)
ProtoInfo, _ = provider(
    doc = "Encapsulates information provided by a proto_library",
    fields = {
        "direct_sources": "Direct .proto sources",
        "transitive_sources": "Transitive .proto sources",
    },
    init = _create_proto_info,
)

def test_function():
    """Test that we can use the ProtoInfo provider."""
    # This should work since ProtoInfo is the first element of the tuple
    info = ProtoInfo(
        direct_sources = [],
        transitive_sources = [],
    )
    return info
