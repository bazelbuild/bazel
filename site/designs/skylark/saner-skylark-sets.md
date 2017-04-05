---
layout: contribute
title: Saner Skylark Sets
---

#  Design Document: Saner Skylark Sets

### (Sacrificing Superfluous Safety)

**Design documents are not descriptions of the current functionality of Skylark.
Always go to the documentation for current information.**


**Status:** Draft

**Author:** [Dmitry Lomov](mailto:dslomov@google.com)

**Design document published**: 2016-07-25

## Motivation

NestedSets (an implementation used in Skylark `set` data type) is an essential
data structure for passing transitive cumulative data during Bazel's analysis.
The reason for that is its memory efficiency: a union of two sets carries a
constant memory overhead (compare that to lists, for example, where a union of
two lists has an O(length of original lists) memory overhead).

When Bazel builds a dependency graph, all data returned by providers coming from
rule implementation stays in memory roughly throughout the lifetime of a Bazel
server. If providers use lists for their cumulative data, the total amount of
memory consumed by them will grow as O(N^2) where N is the number of nodes in
the build graph. Sets are the only data structure available in Skylark that can
reduce that amount to O(N).

However, sets in Skylark suffer from several deficiencies, which preclude their
usage in providers, which leads in turn to performance issues.

*   Skylark sets cannot contain structs (although they can contain tuples, which
    is used sometimes to overcome that limitation:
    [example](https://github.com/bazelbuild/bazel/blob/a48e8e3db5a149777c2887fc7fc572837dd0ac1e/src/test/java/com/google/devtools/build/lib/ideinfo/intellij_info.bzl#L84))
*   Only sets of primitive types (not even tuples) are allowed in rule providers

This document discusses the reasons for these deficiencies and suggests several
ways to resolve them.

## Understanding Current Behavior

### Typing Skylark sets

Skylark sets are, in a certain sense, typed. Heterogeneous Skylark sets (e.g.
ints and strings in the same set) are not allowed. Skylark set carries its
contentType (the type of its elements) with it.

First of all, all elements of Skylark set must be immutable. Sets of sets are
not allowed either.

Second, when a union of skylark sets is computed, their contentTypes are
*intersected* to find a type that contains elements of both sets. Union
operation fails if that intersection is empty.

Type intersection disallows heterogeneous sets since intersection of two
primitive types is empty unless it is one and the same type. However, all tuples
in Skylark have the same type, TUPLE, therefore sets of heterogeneous tuples,
and even tuples of different length, are allowed. Also, since all tuples share
the same type, Skylark set has no information about components of its
constituent tuples.

### Providers and safe values

All values returned as providers from rule and aspect implementations are
required to be *safe*. Safe values are defined as follows (note that the notion
is defined on values, not on types):

*   Primitive safe values are ints, strings, booleans, Files (Artifacts), Labels
    and native providers (TransitiveInfoCollections)
*   Lists of safe values are safe
*   Tuples of safe values are safe
*   Dictionaries are safe if their keys and values are safe
*   Sets are safe if their contentType is that of primitive safe value.

Note that in this entire definition, only the definition of safe sets involves
any types. For all other composite values, their constituents are examined
directly, but in case of sets, this is too expensive (because sets supposedly
hold transitive cumulative information), so their content type is examined
instead.

This poses a problem if we want to allow safe sets of tuples, as set of tuples
forget what are the types of constituents of those tuples.

## What needs to change

As discussed in the "Motivation" section, we need to make sure that Skylark
rules and aspects can use sets to pass information during the build analysis;
that is we need to allow two things:

*   sets should be allowed to contain structs
*   sets of structs should be allowed in rule providers

### Equality and mutability of structs

Allowing sets to contain structs will require equality semantics on structs to
change (from reference to structural equality). We deem the risk of this change
very low; even if structs are compared for equality now in user code, that is
likely a bug, and since structs are immutable, the observed behavior should not
change.

Structs, just like tuples, can be (deeply) mutable and immutable:

```
l = [27, 42, 30]  # This list is mutable (inside a function definition)
t = (l, 42)  # This tuple is mutable, since l can be modified
s = struct(field = l)  # This struct is mutable, since l can be modified
```

Only immutable structs will be allowed as elements of sets, similarly to how
tuples are handled today.

### Provider value safety

To allow sets of structs inside providers, we need to reconcile value safety
check with set typing (Recall that sets forget what constituents their elements
have, so fast safety check is impossible).

#### Alternative #1: Higher-fidelity typing for sets

We can record more precise element types in  sets (not just STRUCT, but the
entire list of fields of those structs). We will need to define what does struct
intersection mean (for example, do we allow sets of structs with disjoint
collections of fields?). If we want to record types for sets of tuples, we will
need to have a very permissive type system to preserve current behavior. For any
restriction we introduce, we need a careful rationale.

#### Alternative #2: Only allow sets of declared providers with type information

If we implement "[Declared providers](/designs/skylark/declared-providers.html)"
proposal and provide more extesive type information on top of that, sets can get
typing information from there.

#### Alternative #3: Abandon safe value check

Looking at all the issues in this document, it appears that safe value check
causes more trouble than the value it brings. The only substantive requirement
for provider values is that they are immutable. Another motivation for a safe
value check was serialization, but if we are serious about serializing
configured targets graph, we will need to learn to serialize arbitrary Skylark
values anyway.
