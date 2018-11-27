---
layout: documentation
title: Depsets
---

# Depsets

[Depsets](lib/depset.html) are a specialized data structure for efficiently
collecting data across a target’s transitive dependencies. Since this use case
concerns the [analysis phase](concepts.md#evaluation-model), depsets are useful
for authors of rules and aspects, but probably not macros.

The main feature of depsets is that they support a time- and space-efficient
merge operation, whose cost is independent of the size of the existing contents.
Depsets also have well-defined ordering semantics.

Example uses of depsets include:

*   storing the paths of all object files for a program’s libraries, which can
    then be passed to a linker action

*   for an interpreted language, storing the transitive source files that will
    be included in an executable's runfiles

If you don't need the merge operation, consider using another type, such as
[list](lib/list.html) or [dict](lib/dict.html).

* ToC
{:toc}

## Full example


This example is available at
[https://github.com/bazelbuild/examples/tree/master/rules/depsets](https://github.com/bazelbuild/examples/tree/master/rules/depsets).

Suppose we have a hypothetical interpreted language Foo. In order to build each
`foo_binary` we need to know all the `*.foo` files that it directly or indirectly
depends on.

```python
# //depsets:BUILD

load(":foo.bzl", "foo_library", "foo_binary")

# Our hypothetical Foo compiler.
py_binary(
    name = "foocc",
    srcs = ["foocc.py"],
)

foo_library(
    name = "a",
    srcs = ["a.foo", "a_impl.foo"],
)

foo_library(
    name = "b",
    srcs = ["b.foo", "b_impl.foo"],
    deps = [":a"],
)

foo_library(
    name = "c",
    srcs = ["c.foo", "c_impl.foo"],
    deps = [":a"],
)

foo_binary(
    name = "d",
    srcs = ["d.foo"],
    deps = [":b", ":c"],
)
```

```python
# //depsets:foocc.py

# "Foo compiler" that just concatenates its inputs to form its output.
import sys

if __name__ == "__main__":
  assert len(sys.argv) >= 1
  output = open(sys.argv[1], "wt")
  for path in sys.argv[2:]:
    input = open(path, "rt")
    output.write(input.read())
```

Here, the transitive sources of the binary `d` are all of the `*.foo` files in
the `srcs` fields of `a`, `b`, `c`, and `d`. In order for the `foo_binary`
target to know about any file besides `d.foo`, the `foo_library` targets need to
pass them along in a provider. Each library receives the providers from its own
dependencies, adds its own immediate sources, and passes on a new provider with
the augmented contents. The `foo_binary` rule does the same, except that instead
of returning a provider, it uses the complete list of sources to construct a
command line for an action.

Here’s a complete implementation of the `foo_library` and `foo_binary` rules.

```python
# //depsets/foo.bzl

# A provider with one field, transitive_sources.
FooFiles = provider(fields = ["transitive_sources"])

def get_transitive_srcs(srcs, deps):
  """Obtain the source files for a target and its transitive dependencies.

  Args:
    srcs: a list of source files
    deps: a list of targets that are direct dependencies
  Returns:
    a collection of the transitive sources
  """
  return depset(
        srcs,
        transitive = [dep[FooFiles].transitive_sources for dep in deps])

def _foo_library_impl(ctx):
  trans_srcs = get_transitive_srcs(ctx.files.srcs, ctx.attr.deps)
  return [FooFiles(transitive_sources=trans_srcs)]

foo_library = rule(
    implementation = _foo_library_impl,
    attrs = {
        "srcs": attr.label_list(allow_files=True),
        "deps": attr.label_list(),
    },
)

def _foo_binary_impl(ctx):
  foocc = ctx.executable._foocc
  out = ctx.outputs.out
  trans_srcs = get_transitive_srcs(ctx.files.srcs, ctx.attr.deps)
  srcs_list = trans_srcs.to_list()
  ctx.actions.run(executable = foocc,
                  arguments = [out.path] + [src.path for src in srcs_list],
                  inputs = srcs_list + [foocc],
                  outputs = [out])

foo_binary = rule(
    implementation = _foo_binary_impl,
    attrs = {
        "srcs": attr.label_list(allow_files=True),
        "deps": attr.label_list(),
        "_foocc": attr.label(default=Label("//depsets:foocc"),
                             allow_files=True, executable=True, cfg="host")
    },
    outputs = {"out": "%{name}.out"},
)
```

You can test this by copying these files into a fresh package, renaming the
labels appropriately, creating the source `*.foo` files with dummy content, and
building the `d` target.

## Description and operations

Conceptually, a depset is a directed acyclic graph (DAG) that typically looks
similar to the target graph. It is constructed from the leaves up to the root.
Each target in a dependency chain can add its own contents on top of the
previous without having to read or copy them.

Each node in the DAG holds a list of direct elements and a list of child nodes.
The contents of the depset are the transitive elements, i.e. the direct elements
of all the nodes. A new depset can be created using the
[depset](lib/globals.html#depset) constructor: it accepts a list of direct
elemens and another list of child nodes.

```python
s = depset(["a", "b", "c"])
t = depset(["d", "e"], transitive = [s])

print(s)    # depset(["a", "b", "c"])
print(t)    # depset(["d", "e", "a", "b", "c"])
```

To retrieve the contents of a depset, use the
[to_list()](lib/depset.html#to_list) method. It returns a list of all transitive
elements, not including duplicates. There is no way to directly inspect the
precise structure of the DAG, although this structure does affect the order in
which the elements are returned.

```python
s = depset(["a", "b", "c"])

print("c" in s.to_list())              # True
print(s.to_list() == ["a", "b", "c"])  # True
```

The allowed items in a depset are restricted, just as the allowed keys in
dictionaries are restricted. In particular, depset contents may not be mutable.

Depsets use reference equality: a depset is equal to itself, but unequal to any
other depset, even if they have the same contents and same internal structure.

```python
s = depset(["a", "b", "c"])
t = s
print(s == t)  # True

t = depset(["a", "b", "c"])
print(s == t)  # False

d = {}
d[s] = None
d[t] = None
print(len(d))  # 2
```

To compare depsets by their contents, convert them to sorted lists.

```python
s = depset(["a", "b", "c"])
t = depset(["c", "b", "a"])
print(sorted(s.to_list()) == sorted(t.to_list()))  # True
```

There is no ability to remove elements from a depset. If this is needed, you
must read out the entire contents of the depset, filter the elements you want to
remove, and reconstruct a new depset. This is not particularly efficient.

```python
s = depset(["a", "b", "c"])
t = depset(["b", "c"])

# Compute set difference s - t. Precompute t.to_list() so it's not done
# in a loop, and convert it to a dictionary for fast membership tests.
t_items = {e: None for e in t.to_list()}
diff_items = [x for x in s.to_list() if x not in t_items]
# Convert back to depset if it's still going to be used for merge operations.
s = depset(diff_items)
print(s)  # depset(["a"])
```

## Order

The `to_list` operation performs a traversal over the DAG. The kind of traversal
depends on the *order* that was specified at the time the depset was
constructed. It is useful for Bazel to support multiple orders because sometimes
tools care about the order of their inputs. For example, a linker action may
need to ensure that if `B` depends on `A`, then `A.o` comes before `B.o` on the
linker’s command line. Other tools might have the opposite requirement.

Three traversal orders are supported: `postorder`, `preorder`, and
`topological`. The first two work exactly like [tree
traversals](https://en.wikipedia.org/wiki/Tree_traversal#Depth-first_search)
except that they operate on DAGs and skip already visited nodes. The third order
works as a topological sort from root to leaves, essentially the same as
preorder except that shared children are listed only after all of their parents.
Preorder and postorder operate as left-to-right traversals, but note that within
each node direct elements have no order relative to children. For topological
order, there is no left-to-right guarantee, and even the
all-parents-before-child guarantee does not apply in the case that there are
duplicate elements in different nodes of the DAG.

```python
# This demonstrates different traversal orders.

def create(order):
  cd = depset(["c", "d"], order = order)
  gh = depset(["g", "h"], order = order)
  return depset(["a", "b", "e", "f"], transitive = [cd, gh])

print(create("postorder").to_list())  # ["c", "d", "g", "h", "a", "b", "e", "f"]
print(create("preorder").to_list())   # ["a", "b", "e", "f", "c", "d", "g", "h"]
```

```python
# This demonstrates different orders on a diamond graph.

def create(order):
  a = depset(["a"], order=order)
  b = depset(["b"], transitive = [a], order = order)
  c = depset(["c"], transitive = [a], order = order)
  d = depset(["d"], transitive = [b, c], order = order)
  return d

print(create("postorder").to_list())    # ["a", "b", "c", "d"]
print(create("preorder").to_list())     # ["d", "b", "a", "c"]
print(create("topological").to_list())  # ["d", "b", "c", "a"]
```

Due to how traversals are implemented, the order must be specified at the time
the depset is created with the constructor’s `order` keyword argument. If this
argument is omitted, the depset has the special `default` order, in which case
there are no guarantees about the order of any of its elements (except that it
is deterministic).

For safety, depsets with different orders cannot be merged with the `+` operator
unless one of them uses the default order; the resulting depset’s order is the
same as the left operand. Note that when two depsets of different order are
merged in this way, the child may appear to have had its elements rearranged
when it is traversed via the parent. **The `+` operator is deprecated, anyway;
use the `transitive` argument instead.**

## Performance

To see the motivation for using depsets, consider what would have happened if we
had implemented `get_transitive_srcs()` without them. A naive way of writing
this function would be to collect the sources in a list.

```python
def get_transitive_srcs(srcs, deps):
  trans_srcs = []
  for dep in deps:
    trans_srcs += dep[FooFiles].transitive_sources
  trans_srcs += srcs
  return trans_srcs
```

However, this does not take into account duplicates, so the source files for `a`
will appear twice on the command line and twice in the contents of the output
file.

The next alternative is using a general set, which can be simulated by a
dictionary where the keys are the elements and all the keys map to `True`.

```python
def get_transitive_srcs(srcs, deps):
  trans_srcs = {}
  for dep in deps:
    for file in dep[FooFiles].transitive_sources:
      trans_srcs[file] = True
  for file in srcs:
    trans_srcs[file] = True
  return trans_srcs
```

This gets rid of the duplicates, but it makes the order of the command line
arguments (and therefore the contents of the files) unspecified, although still
deterministic.

Moreover, both this approach and the list-based one are asymptotically worse
than the depset-based approach. Consider the case where there is a long chain of
dependencies on Foo libraries. Processing every rule requires copying all of the
transitive sources that came before it into a new data structure. This means
that the time and space cost for analyzing an individual library or binary
target is proportional to its own height in the chain. For a chain of length n,
foolib_1 ← foolib_2 ← … ← foolib_n, the overall cost is effectively the
[triangle sum](https://en.wikipedia.org/wiki/Triangular_number) 1 + 2 + … + n,
which is O(n^2). This cost is wasteful because the library rule’s behavior is
not actually affected by the transitive sources.

Generally speaking, depsets should be used whenever you are accumulating more
and more information through your transitive dependencies. This helps ensure
that your build scales well as your target graph grows deeper. The exact
advantage will depend on how deep the target graph is and how many elements per
target are added.

To actually get the performance advantage, it’s important to not retrieve the
contents of the depset unnecessarily in library rules. One call to `to_list()`
at the end in a binary rule is fine, since the overall cost is just O(n). It’s
when many non-terminal targets try to call `to_list()` that we start to get into
quadratic behavior.

## Upcoming changes

The API for depsets is being updated to be more consistent. Here are some recent
and/or upcoming changes.

*   When it's necessary to retrieve a depset's contents, this should be done by
    explicitly converting the depset to a list via its `to_list()` method. Do
    not iterate directly over the depset itself; direct iteration is deprecated
    and will be removed. For example, don't use `list(...)`, `sorted(...)`, or
    other functions expecting an iterable, on depsets. The rationale of this
    change is that iterating over depsets is generally expensive, and expensive
    operations should be made obvious in code.

*   Depset elements currently must have the same type, e.g. all ints or all
    strings. This restriction will be lifted.

*   A merge operation should be done by using the `transitive` argument in the
    depset constructor. All other methods (`|` and `+` operators, and the
    `union` method) are deprecated and will be going away.
