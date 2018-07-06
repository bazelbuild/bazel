load(":inner_dep.bzl", "inner_rule_impl", "prep_work")

prep_work()

my_rule_impl = inner_rule_impl
