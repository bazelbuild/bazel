---
layout: contribute
title: Implementing Beautiful Error Messages (Loading Phase)
---

# Design Document: Implementing Beautiful Error Messages (Loading Phase)

**Design documents are not descriptions of the current functionality of Bazel.
Always go to the documentation for current information.**


**Status**: unimplemented

**Author**: [laurentlb@google.com](mailto:laurentlb@google.com)

**Design document published**: 26 May 2016

**Related**: ["Beautiful error messages"](/designs/2016/05/23/beautiful-error-messages.html)

## Introduction

This is a followup to the document ["Beautiful error messages"](/designs/2016/05/23/beautiful-error-messages.html).

The purpose of this document is to outline a design for some plumbing that will
allow the sort of errors described in that document to be emitted by Blaze for
loading time `BUILD` file errors.

## Review: What needs to be done

In the ["Beautiful error messages"](/designs/2016/05/23/beautiful-error-messages.html)
document, four characteristics of error messages
are enumerated:

1.  **Context**: The erroneous text in the `BUILD` file in question is shown,
    with a caret pointing at the exact expression in question.

2.  **Colors**: Everyone loves colors.

3.  **Suggestions**: A guess of how the error should be fixed is shown.

4.  **Links**: Documentation is referenced directly in error messages.

This document covers (1) and (3).

## Current Error Infrastructure

In the loading phase, Bazel parses `BUILD` files into a tree of [ASTNode]
(https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/syntax/ASTNode.java)
instances, with a [BuildFileAST]
(https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/syntax/BuildFileAST.java)
at the root and `Statement` instances at the leaves. Each statement implements
doExec, which can throw an [EvalException]
(https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/syntax/EvalException.java).
That exception is translated to the error printed to the terminal.

An `EvalException` encapsulates the information given in an error. Fleshing out
the contents of this exception type is a good starting point for implementing
new error features.

## Implementation: Context

An `EvalException` contains a reference to a `Location`, which encapsulates the
line/character information currently specified in a Bazel error.

Generally, an `ASTNode` will have a location instance that is populated by the
parser as it moves from token to token ([example]
(https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/syntax/Parser.java#L607)).
Right now, since the location only contains pure syntactic information, the
parser [calls into the lexer]
(https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/syntax/Parser.java#L413)
to create the location instance. While the parser perhaps *could* also encode
AST information into the `Location`, that shouldn’t be necessary to provide
"context" in the form of a printed line and a carat. It seems that the
lexer maintains `BUILD` file info as a buffer, and should be able to
parameterize `Location` instances with the actual contents of the line in
question. If this is true, then implementing (1) above shouldn’t involve much
more than fleshing out [LocationInfo]
(https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/syntax/Lexer.java#L72).

## Implementation: Suggestions

While data tracked by the lexer should be sufficient to encode an offending
line from a `BUILD` file into an error, providing suggestions will probably
require semantic information only retrievable from the parsed AST. Furthermore,
we require a mechanism in `EvalException` to perform computation on AST data in
order to generate suggestions.

It seems an unlikely solution to encode AST information in `Location` instances,
since those instances are produced by the parser before the AST, or even the AST
node in question, is necessarily complete.

Instead, here are a couple proposals:

1. An abstract subclass of `EvalException` (e.g. `ContextualEvalException`)
that knows how to create an error message with suggestions given unimplemented
suggestion generation logic.

2. A further group of exceptions that are parameterized with a particular sort
of `ASTNode` and know how to generate suggestions. As an example, this
exception type could report typos in rule names.

```java
public class IdentifierEvalException extends ContextualEvalException {
 public static final Map<String, String> TYPOS = Map.of(
    "cclibrary", "cc_library",
    ect...
  )
  public IdentifierEvalException(Identifier identifier, Location loc) {...}
  @Override
  protected Suggestion generateSuggestion() {
    if (TYPOS.keySet().contains(identifier.getName()) {
      return TypoSuggestion(identifier, TYPOS.get(identifier.getName()));
    }
  }
}
```

Once the plumbing around (1) is in place, we can add subclasses at our leisure
to provide suggestions.

Furthermore, a `ContextualEvalException` can be made to have enough information
to not only provide suggestions, but also to return context-aware error
messages. Consider this example, from the
["Beautiful error messages"](/designs/2016/05/23/beautiful-error-messages.html)
document:

```python
my_obj = select({
  ":something": [1],
  "other": [2],
})

t = [x for x in my_obj]
```

<pre>
<font color="red">ERROR:</font> /test/BUILD:6:5: type 'select' is not iterable.
</pre>
<pre>
<font color="red">
ERROR: /test/BUILD:6:16:</font> <b>my_obj of type 'select' is not iterable.</b>
You can iterate only on strings, lists, tuples or dicts.
t = [x for x in my_obj]
                ^
Related documentation: http://documentation#select
</pre>

We can imagine a `NotIterableEvalException` that knows not only about the type
`select`, but is also parameterized with the erroneous expression ```my_objc```.

## Problem: Serialization

The above proposal hinges on the ability to store a file pointer in the
`Location` object, to be dereferenced at error-time to obtain the entire `BUILD`
file. This opens the door to some issues:

1. A `Location` instance is stored for every node in the parse tree of every
`BUILD` file. Even a file pointer in each `Location` may have substantial
memory/speed impact.
<p>This impact is easily measurable and likely tolerable in order to achieve
better error messages. However, it is clear that storing anything much larger
than a file pointer (like a fragment of the file, or the file itself) in each
`Location` would be untenable.

2. The `Location` object is serialized, since the AST is part of a `SkyValue`.
This pointer, then, must be serializable.
<p>This in particular is troubling because, unlike the java heap in a local
Bazel execution, the `SkyValue` containing the AST does not necessarily have the
`BUILD` file. However, it is not clear to me that this means the file must be
copied into each `Location` object. The AST presumably resides in a single
`SkyValue` - one copy of the file in that `SkyValue`, with a pointer to that
file in the `Location`, would be sufficient, it seems.
<p>The nature of the `SkyValue` that contains that AST must be determined,
thinking about if and how to embed the `BUILD` file into that `SkyValue`, and
strategizing about a good serialization for a `Location` object that
contains a file pointer.
