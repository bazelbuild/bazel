---
layout: documentation
title: Extensions - Backward compatibility
---
# Backward compatibility

Bazel is still in Beta and we are going to do breaking changes. As we make
changes and polish the extension mechanism, old features may be removed and new
features that are not backwards-compatible may be added.

Each release, new incompatible changes will be behind a flag with its default
value set to `false`. In later releases, the flag will be enabled by default, or
the flag will be removed entirely.

To check if your code will be compatible with future releases:

*   build your code with the flag `--all_incompatible_changes`, or
*   use boolean flags to enable/disable specific incompatible changes.

This following are the planned incompatible changes that are implemented and
guarded behind flags.

## Set constructor

We are removing the `set` constructor. Use `depset` instead. `set` and `depset`
are equivalent, you just need to do search and replace to update the old code.

We are doing this to reduce confusion between the specialized
[depset](depsets.md) data structure and Python's set datatype.

*   Flag: `--incompatible_disallow_set_constructor`
*   Default: `false`


## Keyword-only arguments

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
*   Default: `false`


## Mutating `+=`

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


## Dictionary concatenation

We are removing the `+` operator on dictionaries. This includes the `+=` form
where the left-hand side is a dictionary. This is done to improve compatibility
with Python. A possible workaround is to use the `.update` method instead.

*   Flag: `--incompatible_disallow_dict_plus`
*   Default: `false`


## Load argument is a label

Historically, the first argument of `load` could be a path with an implicit
`.bzl` suffix. We are going to require that all `load` statements use the label
syntax.

``` python
load("/path/foo", "var")  # deprecated
load("//path:foo.bzl", "var")  # recommended
```

*   Flag: `--incompatible_load_argument_is_label`
*   Default: `false`


## Top level `if` statements

This change forbids `if` statements at the top level of `.bzl` files (they are
already forbidden in `BUILD` files). This change ensures that every global
value has a single declaration. This restriction is consistent with the idea
that global values cannot be redefined.

*   Flag: `--incompatible_disallow_toplevel_if_statement`
*   Default: `true`


## Comprehensions variables

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
*   Default: `false`


## Depset is no longer iterable

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


## String is no longer iterable

When the flag is set to true, `string` objects are not treated as iterable. This
affects `for` loops and many functions, e.g. `list`, `tuple`, `min`, `max`,
`sorted`, `all`, and `any`. String iteration has been a source of errors and
confusion, such as this error:

``` python
def my_macro(name, srcs):
  for src in srcs:
    # do something with src

my_macro("foo")  # equivalent to: my_macro(["f", "o", "o"])
```

String indexing and `len` are still allowed. If you need to iterate over a
string, you may explicitly use:

``` python
my_string="hello world"
for i in range(len(my_string)):
  char = my_string[i]
  # do something with char
```

*   Flag: `--incompatible_string_is_not_iterable`
*   Default: `false`


## Dictionary literal has no duplicates

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
*   Default: `false`


## Checked arithmetic

When set, arithmetic operations (`+`, `-`, `*`) will fail in case of overflow.
All integers are stored using signed 32 bits.

*   Flag: `--incompatible_incompatible_checked_arithmetic`
*   Default: `false`
