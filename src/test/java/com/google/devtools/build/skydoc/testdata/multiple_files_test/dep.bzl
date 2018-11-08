load(":testdata/multiple_files_test/inner_dep.bzl", "inner_rule_impl", "prep_work")

def some_cool_function(name, srcs = [], beef = ""):
    """A pretty cool function. You should call it.

    Args:
      name: Some sort of name.
      srcs: What sources you want cool stuff to happen to.
      beef: Your opinion on beef.
    """
    print(name, srcs, beef)

prep_work()

my_rule_impl = inner_rule_impl
