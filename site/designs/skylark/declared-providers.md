---
layout: contribute
title: Declared Providers
---

# Design Document: Declared Providers

**Design documents are not descriptions of the current functionality of Skylark.
Always go to the documentation for current information.**


**Status:** Approved

**Authors:** [Dmitry Lomov](mailto:dslomov@google.com),
[Laurent Le Brun](mailto:laurentlb@google.com)

**Design document published**: 2016-06-06

## Motivation

Skylark rules use simple Skylark structs as their providers. Skylark providers
are identified as simple names, such as 'java' or 'files'. This approach has the
advantage of simplicity, but as the number and complexity of Skylark rules grow,
we run into engineering scalability problems:

*   Using simple names for providers might lead to name conflicts (when
    unrelated rules call their providers the same simple name).
*   There is no clear formal way to add documentation for those providers; if
    any, the documentation is in prose in rule's doc comment, where it tends to
    become obsolete/incomplete; most existing providers have no documentation
    explaining their contracts at all.
*   It’s hard to know which fields to expect in a provider.
*   It’s hard to know which rule can depend on which rule.

## Goals

*   Solve name-conflict problem for providers
*   Allow to specify providers in Skylark rules with the same level of
    robustness as other components of the language, such as rules and aspects
*   Enable the same or better documentability of Skylark providers as native
    providers allow
*   Improve providers interoperability with native code.

## Proposal

We propose a redesign of how Skylark rules deal with providers to address the
above concerns. The redesign can occur in stages; those stages represent
implementation stages, but allow Skylark users to gradually opt for "more and
more engineering" as their custom rules progress from a small project on the
side to a public release.

Our proposal is backwards compatible with the existing providers in Bazel and
allows easy, gradual piecemeal replacement of them.

### Stage 1: Solving the Naming Problem

Under the new proposal, a minimum implementation of a custom provider looks like
this:

```
# rust.bzl

# Introduces a provider. `rust_provider` is now both a function
# that can be used to construct the provider instance,
# and a "symbol" that can be used to access it.
rust_provider = provider()

def _impl(ctx):
  # `rust_provider` is used as struct-like constructor
  # it accepts the same arguments as a standard `struct` function
  rust = rust_provider(defines = "-DFOO", ...)
  # return value of rule implementation function
  # is just a list of providers; their "names" are specified
  # by their constructors, see below
  return [ctx.provider(files = ...), rust]
rust_library = rule(implementation = _impl,
  # Optional declaration; the rule MUST provide all the
  # providers in this list
  providers = [rust_provider])
```

```
# Example of how to access the provider

load(":rust.bzl", "rust_provider")

def _impl(ctx):
  dep = ctx.attr.deps[0] # Target object
  # `rust_provider` is used as a key to access a particular
  # provider
  defines = dep[rust_provider].defines ...
```

#### The provider function

*   We introduce two new kinds of Skylark values, a *provider declaration* and a
    *provider*.
*   Provider declaration (`rust_provider` in the example) is created using the
    `provider` function.
*   Provider declaration can be used to construct a *provider* (`rust` in the
    example). Provider is a struct-like Skylark value, with the only difference
    that every provider is associated with its declaration (it is a different
    type). Arguments of a provider declaration when used as a function are
    exactly the same as that of a built-in `struct` function.
*   [Target](http://www.bazel.build/docs/skylark/lib/Target.html) objects become
    dictionaries of providers indexed by their declaration. Bracket notation can
    be used to retrieve a particular provider. Thus, provider declarations are
    [symbol-like](https://developer.mozilla.org/en/docs/Web/JavaScript/Reference/Global_Objects/Symbol)
    values.
*   Providers can be private to an extension file; in that case the provider
    cannot be accessed outside that file.

#### Default providers (ctx.provider)

There is a set of default providers (`files`, `runfiles`, `data_runfiles`,
`executable`, `output_groups`, etc.). We group them in a single provider,
`ctx.provider`:

```
defaults = ctx.provider(files = set(), runfiles = ...)
```

The current set of APIs on Target objects that access these providers
(`target.files`, `target.output_group("name")` etc.) will continue to work.

#### Return value

The implementation function can return either a provider, or a list of
providers. It is an error to return two providers of the same type.

```
return [defaults, rust, cc]
return ctx.provider(files = set())
```

#### Declaring providers returned by a rule

Users need to know which rules provide which providers. This is important for
documentation and for knowing which dependencies are allowed (e.g. we want to
find easily what can go in the deps attribute of cc_library).

We allow rules to declare the providers they intend to return with a `providers`
argument of a
<code>[rule](http://www.bazel.build/docs/skylark/lib/globals.html#rule)</code>
function. It is an error if the rule implementation function does not return all
the providers listed in `providers`. It may however return additional providers.

```
rust_provider = provider()

rust_library = rule(implementation = _impl,
  # Optional declaration; the rule MUST provide all the
  # providers in this list
  providers = [rust_provider])
```

#### Migration path and support for "legacy" providers

To support current model of returning providers, where they are identified by a
simple name, we continue to allow providers name in the return struct:

```
def _impl(ctx):
  ...
  return struct(
    legacy_provider = struct(...),
    files = set(...),
    providers = [rust])
```

This also works for “default” providers, such as “files”, “runfiles” etc.
However if one of those legacy names is specified, it is an error to have
ctx.provider instance in the list of `providers`.

We also allow returning a declared provider both directly and with a simple
name:

```
def _impl(ctx):
  ...
  return struct(rust = rust, providers = [rust])
```

This allows the rules to mix old and new style, and migrate rule definition to a
new style without changing all the uses of that rule.

Old-style providers with simple names can still be accessed with dot-notation on
Target object, so all of the following is valid.

Old-style usage:

*   `target.rust` (=> rust)
*   `getattr(target, "rust")` (=> rust)
*   `hasattr(target, "rust")` (=> True)

New-style usage:

*   `target[rust_provider]` (=> rust)
*   `rust_provider in target` (=> True)
*   `target.keys` (=> [rust_provider])
*   `target.values` (=> [rust])
*   `target.items` (=> [(rust_provider, rust)])

#### type function

Type function on providers returns a string `"provider"`. Type function on a
provider instance returns a string `"struct"`.

### Stage 2: Documentation and Fields

Provider declarations are a convenient place to add more annotations to
providers. We propose 2 specific things there:

```
rust_provider = provider(
  doc = "This provider contains Rust information ...",
  fields = ["defines", "transitive_deps"]
)
```

This specifies documentation for the provider and a list of fields that the
provider can have.

If `fields` argument is present, extra, undeclared fields are not allowed.

Both `doc` and `fields` arguments to `provider` function are optional.

`fields` argument can also be a dictionary (from string to string), in that case
the keys are names of fields, and the values are documentation strings about
individual fields

```
rust_provider = provider(
  doc = "This provider contains Rust information ...",
  fields = {
    "defines": "doc for define",
    "transitive_deps": "doc for transitive deps,
  })
```

### Native Providers

Providers (as Skylark values) can be also declared natively. A set of
annotations can be developed to facilitate declaring them with little effort.

As a strawman example:

```
/**
 * Hypothetical implementation of Skylark provider value (result of
 * provider(..) function.
 */
class SkylarkProviderValue extends SkylarkValue {
  ...
  /**
   * Creates a SkylarkProviderValue for a native provider
   * `native` must be annotated with @SkylarkProvider annotation.
   * Field accessors and constructor function appear magically.
   */
  static <T> SkylarkProviderValue forNative(Class<T> native) { ... }
}

@SkylarkProvider(builder = Builder.class)
// A class with this annotation can be used as provider declaration
class rustProvider implements TransitiveInfoProvider {
  @SkylarkProviderField(doc = ...)
  // Skylark name is 'defines'
  String getDefines() { ... }

  @SkylarkProviderField(doc = ...)
  // Skylark name is 'transitive_deps'
  NestedSet<Artifact> getTransitiveDeps() { ... }

  @SkylarkProviderField(doc = ...)
  // Not allowed, the set of types exposed to Skylark is restricted
  DottedVersion getVersion() { ... }

  // Automatically used to provide an implementation for
  // construction function.
  static class Builder {
    // a setter for 'defines' field, based on name.
    void setDefines(String defines) { ... }
    // a setter for 'transitive_deps' field, based on name.
    void setTransitiveDeps(...) {...}
    rustProvider build() { ... }
  }
}
```
