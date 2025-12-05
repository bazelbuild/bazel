"""Test file that uses depset to verify builtin functions work correctly."""

# Test depset creation at module load time (like rules_go does)
_EMPTY_DEPSET = depset([])
_STRING_DEPSET = depset(["a", "b", "c"])

def depset_function(items):
    """A function that creates and returns a depset.

    Args:
        items: A list of items to put in the depset

    Returns:
        A depset containing the items
    """
    return depset(items)

def depset_with_order(items, order = "stable"):
    """Creates a depset with a specific order.

    Args:
        items: Items to include
        order: Order type (stable, preorder, postorder, topological)

    Returns:
        An ordered depset
    """
    return depset(items, order = order)

def depset_with_transitive(direct, transitive):
    """Creates a depset with direct and transitive items.

    Args:
        direct: Direct items
        transitive: List of depsets to include transitively

    Returns:
        A depset combining direct and transitive items
    """
    return depset(direct = direct, transitive = transitive)

DepsetInfo = provider(
    doc = "A provider that holds depsets.",
    fields = {
        "files": "A depset of files",
        "empty": "An empty depset",
    },
)

def _depset_rule_impl(ctx):
    """Implementation that uses depsets."""
    files = depset([f.path for f in ctx.files.srcs])
    return [DepsetInfo(files = files, empty = _EMPTY_DEPSET)]

depset_rule = rule(
    implementation = _depset_rule_impl,
    doc = "A rule that uses depsets.",
    attrs = {
        "srcs": attr.label_list(
            doc = "Source files",
            allow_files = True,
        ),
    },
)
