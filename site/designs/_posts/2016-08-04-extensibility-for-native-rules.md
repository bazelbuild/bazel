---
layout: contribute
title: Extensibility For Native Rules
---

# Design Document: Extensibility For Native Rules

**Design documents are not descriptions of the current functionality of Bazel.
Always go to the documentation for current information.**


**Status**: Reviewed, not yet implemeted

**Author**: [Dmitry Lomov](mailto:dslomov@google.com)

**Design document published**: 04 August 2016

## Motivation

There is a number of requests that require Skylark API to access functionality
of native rules from Skylark rules. Typical scenarios can be illustrated by the
following "sandwich":

```python
bread_library(name = "top", …)
java_library(name = "meat", deps = [":top", …], …)
bread_library(name = "bottom", deps = [":meat", …])
```

Here bread\_library is a rule written in Skylark. Here we need three things:

* implementation of bread\_library should be able to produce jar files and other
  artifacts in the same way as native java\_library rule would; in other words
  the implementation should be able to delegate to the a native implementation
* java\_library should allow depending on bread\_library; importantly, that
  dependence should be *meaningful*, that is if bread\_library produces Java
  artifacts, such as jar files, java\_library should be able to compile against
  those like it would against a java\_library dependency
* bread\_library should be able to depend on java\_library; it should be able to
  access all information it needs; when delegating to native implementation,
  there should be a simple way to pass information from a dependency to to the
  native implementation

In this document we present some ideas about what how this all might look like,
and suggest some practical steps we can take to get there.

## Extensible Native Rules

This proposal assumes [Declared Providers
](/designs/skylark/declared-providers.html)
are implemented. Here is how the implementation of bread\_library might look
like:

```python
# Implementation of a rule that transpiles to Java and invokes a native
# compilation
def _bread_library_impl(ctx):
    bread_sources = [f for src in ctx.attrs.src for f in src.files]
    generated_java_files = _invoke_bread_transpiler(ctx, bread_sources)
    # lang.java.provider is a declared provider for Java
    java_deps = [target[lang.java.provider] for target in ctx.attrs.deps]
    # create a native compilation action
    java_p = lang.java.compile(ctx,
                srcs = generated_java_files,
                # information about dependencies is just a lang.java.provider
                deps = java_deps,
                ...)
    # java_p is a lang.java.provider representing the result of compilation
    # action we return that provider and immediately java_libary rule can depend
    # on us
    return [java_p, ...]

# Implementation of a rule that compiles to JVM bytecode directly
def _scala_library_impl(ctx):
    # collect dependency jars to pass to the compile action
    dep_jars = [dep[java.lang.provider].jar for dep in ctx.attrs.deps]
    jar_output = ctx.new_file(...)
    ... construct compilation actions ...
    # build a provider that passes all transitive information
    transitive_p = lang.java.transitive(
                      [dep[java.lang.provider] for dep in ctx.attrs.deps])
    java_p = lang.java.provider(
                transitive_p,
                jar = jar_output,
                # update transitive information that we care about
                transitive_jars =
                    transitive_p.transitive_jars | set(jar_output),
                    ... whatever other information is needed ...)
    # return java.lang.provider
    return [java_p, ...]
```

The provider is the glue ("butter") that connects Skylark rules to native rules
and also to the native rule implementations exposed to Skylark. Note how the
native rule implementation (lang.java.compile) both consumes the entire
providers from dependencies and returns the provider that needs to be returned
from the rule. `lang.java.transitive` is a function that passes all the
transitive information correctly from dependencies. The [existing '.java'
provider](http://www.bazel.build/docs/skylark/lib/JavaSkylarkApiProvider.html)
becomes the same thing as lang.java.provider.

Note: for the sake for this document we are placing things in lang.java. There
are other alternatives to this, e.g. "magical" .bzl files from which
java\_provider and java\_compile function are exported.

## How to get there

Our current native rules are not as neat as described above. Making them
extensible in one go is a difficult and long term project (or rather, projects:
one for each language). Here is a suggested steps for extensibility of
particular language implementation (we continue to use Java as a running
example).

### Phase 1: Expose native compilation actions

At this step, lang.java.provider is a *black box*. Skylark rules cannot
construct the lang.java.provider directly: the only way to create it is to
invoke lang.java.compile function.

Native implementations of Java rules are rewritten so that they can link to deps
that return lang.java.provider and that they return lang.java.provider.  The
implementation of the provider can just be a bag of all providers that Java
rules normally return - since that bag is not openable by Skylark, we can
refactor it later without much difficulty .

Native compilation function (java.lang.compile) is pretty much
JavaLibrary.create refactored so that it gets its dependent providers not from
attributes but from a list of bags. JavaLibrary.create just collects the bags
from deps and passes those to that function.

At the end of this phase, implementing code generators (and code generating
aspects) such as java\_proto\_library becomes possible. This also covers many
(most?) use cases where people use macros to delegate to native rule
implementations

No huge refactoring of language rule implementation is needed, but the stage is
set for gradual opening up in the future.

### Phase 1a: Implementing JavaSkylarkApiProvider on top of black box

(Optional) As lang.java.provider is just a bag of existing providers, it is easy
to just implement everything in 'target.java' on top of it, if desired.

### Phase 2: Evolving the API and opening up

The next step in the API evolution is making the black box provider less black.
This means introducing a constructor for lang.java.provider as well as accessors
to fields.

The API can be designed gradually and thoughtfully, only exposing the things we
need and adding carefully: as an example sequence first just java libraries,
then resources, then JNI, then support for tests. Existence of
lang.java.transitive is crucial at this stage as it allows merging of transitive
information from dependencies that is not yet exposed to Skylark.

As API exposure gradually progresses, the exposed Skylark API reaches parity
with internal API.

Through the execution of this phase, more and more use cases are covered, and at
the end the rules are fully extensible.
