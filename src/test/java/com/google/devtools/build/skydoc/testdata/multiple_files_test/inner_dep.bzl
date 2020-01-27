"""A deep dependency file."""

def prep_work():
    """Does some prep work. Nothing to see here."""
    return 1

def inner_rule_impl(ctx):
    _ignore = [ctx]
    return struct()
