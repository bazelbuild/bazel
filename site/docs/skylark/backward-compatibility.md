---
layout: documentation
title: Backward Compatibility
---

# Backward Compatibility

Bazel is still in Beta and new releases may include backward incompatible
changes. As we make changes and polish the extension mechanism, old features
may be removed and new features that are not backward compatible may be added.

Backward incompatible changes are introduced gradually:

1.  The backward incompatible change is introduced behind a flag with its
    default value set to `false`.
2.  In a later release, the flag's default value will be set to `true`. You
    can still use the flag to disable the change.
3.  Then in a later release, the flag will be removed and you will no longer be
    able to disable the change.

To check if your code will be compatible with future releases you can:

*   Build your code with the flag `--all_incompatible_changes`. This flag
    enables all backward incompatible changes, and so you can ensure your code
    is compatible with upcoming changes.
*   Use boolean flags to enable/disable specific backward incompatible changes.

## Current backward incompatible changes

The following are the backward incompatible changes that are implemented and
guarded behind flags in the current release:

*   [Dictionary concatenation](#dictionary-concatenation)
*   [Load must appear at top of file](#load-must-appear-at-top-of-file)
*   [Top level `if` statements](#top-level-if-statements)
*   [Depset is no longer iterable](#depset-is-no-longer-iterable)
*   [Depset union](#depset-union)
*   [String is no longer iterable](#string-is-no-longer-iterable)
*   [New actions API](#new-actions-api)
*   [Glob tracking](#glob-tracking)


### Dictionary concatenation

We are removing the `+` operator on dictionaries. This includes the `+=` form
where the left-hand side is a dictionary. This is done to improve compatibility
with Python. A possible workaround is to use the `.update` method instead.

*   Flag: `--incompatible_disallow_dict_plus`
*   Default: `false`


### Load must appear at top of file

Previously, the `load` statement could appear anywhere in a `.bzl` file so long
as it was at the top level. With this change, for `.bzl` files, `load` must
appear at the beginning of the file, i.e. before any other non-`load` statement.

*   Flag: `--incompatible_bzl_disallow_load_after_statement`
*   Default: `false`


### Top level `if` statements

This change forbids `if` statements at the top level of `.bzl` files (they are
already forbidden in `BUILD` files). This change ensures that every global
value has a single declaration. This restriction is consistent with the idea
that global values cannot be redefined.

*   Flag: `--incompatible_disallow_toplevel_if_statement`
*   Default: `true`


### Depset is no longer iterable

When the flag is set to true, `depset` objects are not treated as iterable. If
you need an iterable, call the `.to_list()` method. This affects `for` loops and
many functions, e.g. `list`, `tuple`, `min`, `max`, `sorted`, `all`, and `any`.
The goal of this change is to avoid accidental iteration on `depset`, which can
be expensive.

``` python
deps = depset()
[x.path for x in deps]  # deprecated
[x.path for x in deps.to_list()]  # recommended

sorted(deps)  # deprecated
sorted(deps.to_list())  # recommended
```

*   Flag: `--incompatible_depset_is_not_iterable`
*   Default: `false`


### Depset union

To merge two sets, the following examples used to be supported, but are now
deprecated:

``` python
depset1 + depset2
depset1 | depset2
depset1.union(depset2)
```

The recommended solution is to use the `depset` constructor:

``` python
depset(transtive=[depset1, depset2])
```

See the [`depset documentation`](depsets.md) for more information.

*   Flag: `--incompatible_depset_union`
*   Default: `false`


### String is no longer iterable

When the flag is set to true, `string` objects are not treated as iterable. This
affects `for` loops and many functions, e.g. `list`, `tuple`, `min`, `max`,
`sorted`, `all`, and `any`. String iteration has been a source of errors and
confusion, such as this error:

``` python
def my_macro(name, srcs):
  for src in srcs:
    # do something with src

# equivalent to: my_macro("hello", ["f", "o", "o", ".", "c", "c"])
my_macro(
  name = "hello",
  srcs = "foo.cc",
)
```

String indexing and `len` are still allowed. If you need to iterate over a
string, you may explicitly use:

``` python
my_string = "hello world"
for i in range(len(my_string)):
  char = my_string[i]
  # do something with char
```

*   Flag: `--incompatible_string_is_not_iterable`
*   Default: `false`


### New actions API

This change removes the old methods for registering actions within rules, and
requires that you use the new methods instead. The deprecated methods and their
replacements are as follows.

*   `ctx.new_file(...)` --> `ctx.actions.declare_file(...)`
*   `ctx.experimental_new_directory(...)` -->
    `ctx.actions.declare_directory(...)`
*   `ctx.action(...)` --> either `ctx.actions.run(...)` or
    `ctx.actions.run_shell(...)`
*   `ctx.file_action(...)` --> `ctx.actions.write(...)`
*   `ctx.empty_action(...)` --> `ctx.actions.do_nothing(...)`
*   `ctx.template_action(...)` --> `ctx.actions.expand_template(...)`

<!-- filler comment, needed by Markdown to separate the lists -->

*   Flag: `--incompatible_new_actions_api`
*   Default: `false`


### Glob tracking

When set, glob tracking is disabled. This is a legacy feature that we expect has
no user-visible impact.

*   Flag: `--incompatible_disable_glob_tracking`
*   Default: `true`

<!-- Add new options here -->
