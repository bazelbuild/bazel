---
layout: documentation
title: Extensions - Backward compatibility
---

# Backward compatibility

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
    enables all backward incomaptible changes, and so you can ensure your code
    is compatible with upcoming changes.
*   Use boolean flags to enable/disable specific backward incompatible changes.

## Current backward incompatible changes

The following are the backward incompatible changes that are implemented and
guarded behind flags in the current release:

*   [Keyword-only arguments](#keyword-only-arguments)
*   [Mutating `+=`](#mutating)
*   [Dictionary concatenation](#dictionary-concatenation)
*   [Load must appear at top of file](#load-must-appear-at-top-of-file)
*   [Load argument is a label](#load-argument-is-a-label)
*   [Top level `if` statements](#top-level-if-statements)
*   [Comprehensions variables](#comprehensions-variables)
*   [Depset is no longer iterable](#depset-is-no-longer-iterable)
*   [String is no longer iterable](#string-is-no-longer-iterable)
*   [Dictionary literal has no duplicates](#dictionary-literal-has-no-duplicates)
*   [New actions API](#new-actions-api)
*   [Checked arithmetic](#checked-arithmetic)

### Keyword-only arguments

Keyword-only parameters are parameters that can be called only using their name.

``` python
def foo(arg1, *, arg2): pass

foo(3, arg2=3)
```

``` python
def bar(arg1, *rest, arg2): pass

bar(3, arg2=3)
```

In both examples, `arg2` must be named at the call site. To preserve syntactic
compatibility with Python 2, we are removing this feature (which we have never
documented).

*   Flag: `--incompatible_disallow_keyword_only_args`
*   Default: `true`


### Mutating `+=`

We are changing `left += right` when `left` is a list. The old behavior is
equivalent to `left = left + right`, which creates a new list and assigns it to
`left`. The new behavior does not rebind `left`, but instead just mutates the
list in-place.

``` python
def fct():
  li = [1]
  alias = li
  li += [2]
  # Old behavior: alias == [1]
  # New behavior: alias == [1, 2]
```

This change makes Skylark more compatible with Python and avoids performance
issues. The `+=` operator for tuples is unaffected.

*   Flag: `--incompatible_list_plus_equals_inplace`
*   Default: `false`


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


### Load argument is a label

Historically, the first argument of `load` could be a path with an implicit
`.bzl` suffix. We are going to require that all `load` statements use the label
syntax.

``` python
load("/path/foo", "var")  # deprecated
load("//path:foo.bzl", "var")  # recommended
```

*   Flag: `--incompatible_load_argument_is_label`
*   Default: `false`


### Top level `if` statements

This change forbids `if` statements at the top level of `.bzl` files (they are
already forbidden in `BUILD` files). This change ensures that every global
value has a single declaration. This restriction is consistent with the idea
that global values cannot be redefined.

*   Flag: `--incompatible_disallow_toplevel_if_statement`
*   Default: `true`


### Comprehensions variables

This change makes list and dict comprehensions follow Python 3's semantics
instead of Python 2's. That is, comprehensions have their own local scopes, and
variables bound by comprehensions are not accessible in the outer scope.

As a temporary measure to help detect breakage, this change also causes
variables defined in the immediate outer scope to become inaccessible if they
are shadowed by any variables in a comprehension. This disallows any uses of the
variable's name where its meaning would differ under the Python 2 and Python 3
semantics. Variables above the immediate outer scope are not affected.

``` python
def fct():
  x = 10
  y = [x for x in range(3)]
  return x
```

The meaning of this program depends on the flag:

 * Under Skylark without this flag: `x` is 10 before the
   comprehension and 2 afterwards. (2 is the last value assigned to `x` while
   evaluating the comprehension.)

 * Under Skylark with this flag: `x` becomes inaccessible after the
   comprehension, so that `return x` is an error. If we moved the `x = 10` to
   above the function, so that `x` became a global variable, then no error would
   be raised, and the returned number would be 10.

In other words, please do not refer to a loop variable outside the list or dict
comprehension.

*   Flag: `--incompatible_comprehension_variables_do_not_leak`
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


### Dictionary literal has no duplicates

When the flag is set to true, duplicated keys are not allowed in the dictionary
literal syntax.

``` python
{"a": 2, "b": 3, "a": 4}  # error
```

When the flag is false, the last value overrides the previous value (so the
example above is equivalent to `{"a": 4, "b": 3}`. This behavior has been a
source of bugs, which is why we are going to forbid it.

If you really want to override a value, use a separate statement:
`mydict["a"] = 4`.

*   Flag: `--incompatible_dict_literal_has_no_duplicates`
*   Default: `true`


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


### Checked arithmetic

When set, arithmetic operations (`+`, `-`, `*`) will fail in case of overflow.
All integers are stored using signed 32 bits.

*   Flag: `--incompatible_checked_arithmetic`
*   Default: `true`


<!-- Add new options here -->
