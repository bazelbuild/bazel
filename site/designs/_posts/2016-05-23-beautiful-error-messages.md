---
layout: documentation
title: Beautiful error messages
---

# Design Document: Beautiful error messages

**Design documents are not descriptions of the current functionality of Bazel.
Always go to the documentation for current information.**


**Status**: Reviewed, not yet implemented

**Author**: laurentlb@google.com

**Reviewers**: ulfjack@google.com

**Design document published**: 23 May 2016

**Related**: [Implementing beautiful error
messages](/designs/2016/05/26/implementing-beautiful-error-messages.html)

## Background

Providing good error messages is important for usability. It is especially
important for new users who try to understand a tool. It can also drive adoption
(as in clang vs gcc).

I suggest we make error messages in Bazel more helpful.

## Other tools

Screenshots below are from Elm and Clang, both have been praised for the quality
of their diagnostics. They are good examples we can draw our inspiration from.

See also [Compiler Errors for Humans]
(http://elm-lang.org/blog/compiler-errors-for-humans), a blog article about the
work that went into Elm.

![Example 1](/assets/error_example_1.png "Example 1")

![Example 2](/assets/error_example_2.png "Example 2")

![Example 3](/assets/error_example_3.png "Example 3")

![Example 4](/assets/error_example_4.png "Example 4")

## What we can improve

*   Users need **context**: Showing exact code in error is useful (with a caret
    ^). Currently location in Bazel error messages is often approximative (e.g.
    column number is often off, we show location of a rule instead of the
    specific attribute).
*   **Colors** make messages easier to understand.
*   **Suggestions** are useful. Bazel should try to guess what the user wanted
    to do, and suggest a fix. A typical example is to detect typos.
*   A **link** can give more more information (e.g. elm can show a link to
    https://github.com/elm-lang/elm-compiler/blob/master/hints/recursive-alias.md).
    If you don’t, users will have to copy error messages and look up on search
    engine.

We have some limitations though:

*   Due to memory constraints, we cannot keep the source file or even the AST
    after loading phase. Maybe code printing can be limited to errors in loading
    phase?
*   Code generated through macros can make error reporting more difficult.

## Examples

In each example below, you’ll find 1. the input code, 2. the current output from
Bazel, 3. a suggested improvement.

These are just suggestions we can iterate on.

***

Input:
<pre><code>my\_obj = select({
    ":something": [1],
    "other": [2],
})
t = [x for x in my\_obj]
</code></pre>

Current:
<pre><code>ERROR: /path/BUILD:6:5: type 'select' is not iterable
</code></pre>
Suggested:
<pre><code>ERROR: /path/BUILD:6:16: <strong>my\_obj of type 'select' is not iterable.</strong> You can iterate only on string, lists, tuples, or dicts.
t = [x for x in my\_obj]
                ^-----
Related documentation: http://www.bazel.build/docs/be/functions.html#select
</code></pre>

***

Input:
<pre><code>t = [x for x in]
</code></pre>
Current:
<pre><code>ERROR: /path/BUILD:1:16: syntax error at ']': expected expression
</code></pre>
Suggested:
<pre><code>ERROR: /path/BUILD:1:16: <strong>Syntax error: expected expression, got ']'.</strong>
t = [x for x in]
               ^
</code></pre>

***

Input:
<pre><code>glob(["*.cc"], excludes = ["foo.cc"])
</code></pre>

Current:
<pre><code>ERROR: /path/BUILD:1:1: unexpected keyword 'excludes' in call to glob(include: sequence of strings, exclude: sequence of strings = [], exclude\_directories: int = 1)
</code></pre>
Suggested:
<pre><code>ERROR: /path/BUILD:1:5: <strong>'excludes' is an invalid keyword argument for the function glob(include, exclude, exclude\_directories).</strong> Did you mean exclude?
glob(["*.cc"], excludes = ["foo.cc"])
               ^-------
               exclude
Related documentation: http://www.bazel.build/docs/be/functions.html#glob
</code></pre>

***

Input:
<pre><code>cclibrary(name = "x")
</code></pre>
Current:
<pre><code>ERROR: /path/BUILD:1:1: name 'cclibrary' is not defined
</code></pre>
Suggested:
<pre><code>ERROR: /path/BUILD:1:1: <strong>Name 'cclibrary' is not defined.</strong> Did you mean cc\_library?
cclibrary(name = "x")
^--------
cc\_library
</code></pre>

***

Input:
<pre><code>cc\_library(
    name = "x",
    deps = ":lib",
)
</code></pre>
Current:
<pre><code>ERROR: /path/BUILD:1:1: //test:x: expected value of type 'list(label)' for attribute 'deps' in 'cc\_library' rule, but got ":lib" (string)
</code></pre>
Suggested:
<pre><code>ERROR: /path/BUILD:3:5: <strong>Expected value of type 'list(string)' for attribute 'deps' in 'cc\_library' rule, but got ":lib" (string). </strong>Did you mean [":lib"]?
    deps = ":lib",
           ^-----
           [":lib"]
Related documentation: http://www.bazel.build/docs/be/c-cpp.html#cc\_library
</code></pre>

***

Input:
<pre><code>VAR = ":lib"
cc\_library(
    name = "x",
    deps = VAR,
)
</code></pre>
Current:
<pre><code>ERROR: /path/BUILD:3:1: //test:x: expected value of type 'list(label)' for attribute 'deps' in 'cc\_library' rule, but got ":lib" (string)
</code></pre>
Suggested:
<pre><code>ERROR: /path/BUILD:5:5: <strong>Expected value of type 'list(string)' for attribute 'deps' in 'cc\_library' rule, but got ":lib" (string).</strong> Did you mean [VAR]?
    deps = VAR,
           ^--
          [VAR]
Related documentation: http://www.bazel.build/docs/be/c-cpp.html#cc\_library
</code></pre>

***

Input:
<pre><code>cc\_library(
    name = "name",
    deps = ["/test/foo.cc"],
)
</code></pre>
Current:
<pre><code>ERROR: /path/BUILD:1:1: //test:name: invalid label '/test/foo.cc' in element 0 of attribute 'srcs' in 'cc\_library' rule: invalid target name '/test/foo.cc': target names may not start with '/'
</code></pre>
Suggested:
<pre><code>ERROR: /path/BUILD:3:13: <strong>Invalid label '/test:foo.cc' in deps. Labels relative to the root start with //.</strong> Did you mean '//test:foo.cc'?
    deps = ["/test:foo.cc"],
            ^-------------
            "//test:foo.cc"
Related documentation: http://www.bazel.build/docs/build-ref.html#labels
</code></pre>

***

Input:
<pre><code>cc\_library(
    name = "name",
    srcs = [":x"],
)
genrule(
    name = "x",
    outs = ["file.ext"],
    cmd = "touch $@",
)
</code></pre>
Current:
<pre><code>ERROR: /path/BUILD:3:12: in srcs attribute of cc\_library rule //test:name: '//test:x' does not produce any cc\_library srcs files (expected .cc, .cpp, .cxx, .c++, .C, .c, .h, .hh, .hpp, .hxx, .inc, .S, .s, .asm, .a, .pic.a, .lo, .pic.lo, .so, .dylib, .o or .pic.o)
</code></pre>
Suggested:
<pre><code>ERROR: /path/BUILD:3:12: <strong>In srcs attribute of //test:name (cc\_library), '//test:x' does not produce any cc\_library srcs files</strong> (expected extension .cc, .cpp, .cxx, .c++, .C, .c, .h, .hh, .hpp, .hxx, .inc, .S, .s, .asm, .a, .pic.a, .lo, .pic.lo, .so, .dylib, .o or .pic.o). <strong>Target //test:x (genrule) generated 'file.ext'</strong>.
Related documentation: http://www.bazel.build/docs/be/c-cpp.html#cc\_library
</code></pre>

***

Input:
<pre><code>cc\_library(
    name = "name",
    deps = ["//base:scheduling\_domain-test"],
)
</code></pre>
Current:
<pre><code>ERROR: /path/BUILD:1:1: in cc\_library rule //test:name: non-test target '//test:name' depends on testonly target '//base:scheduling\_domain-test' and doesn't have testonly attribute set
</code></pre>
Suggested:
<pre><code>ERROR: /path/BUILD:3:5: <strong>In deps attribute of //test:name (cc\_library), '//base:scheduling\_domain-test' (cc\_library) is marked as testonly.</strong> You may either add:
    testonly = 1
to //test:name definition, or remove testonly from //base:scheduling\_domain-test, or remove the dependency.
Related documentation: http://www.bazel.build/docs/be/common-definitions.html#common.testonly
</code></pre>

***

Input:
<pre><code>cc\_library(
    name = "name",
    srcs = ["//base:arena.cc"],
)
</code></pre>
Current:
<pre><code>ERROR: /path/BUILD:1:1: Target '//base:arena.cc' is not visible from target '//test:name'. Check the visibility declaration of the former target if you think the dependency is legitimate
</code></pre>
Suggested:
<pre><code>ERROR: /path/BUILD:3:5: <strong>In srcs attribute of //test:name (cc\_library), '//base:arena.cc' (file) is not visible.</strong> You may change the visibility of the file using exports\_file, or expose the file via a library rule, or remove the dependency.
//base:arena.cc has currently private visibility.
Related documentation: http://www.bazel.build/docs/be/common-definitions.html#common.visibility
</code></pre>

***

Input:
<pre><code>cc\_binary(
    name = "bin",
    deps = [":lib"],
)
cc\_library(
    name = "lib",
    srcs = [":src"],
)
genrule(
    name = "src",
    outs = ["file.cc"],
    cmd = "touch $@",
    tools = [":bin"],
)
</code></pre>
Current:
<pre><code>ERROR: /path/BUILD:1:1: in cc\_binary rule //test:bin: cycle in dependency graph:
    //test:bin
    //test:lib
    //test:src
  \* //test:bin (host)
    //test:lib (host)
    //test:src (host)
  \* //test:bin (host)
</code></pre>
Suggested:
<pre><code>ERROR: /path/BUILD:1:1: <strong>Cycle in dependency graph detected:</strong>
    cc\_binary //test:bin depends on (via deps):
    cc\_library //test:lib depends on (via srcs):
    genrule //test:src depends on (via tools):
  <strong>\* cc\_binary //test:bin</strong> [host] depends on (via deps):
  <strong>| cc\_library //test:lib</strong> [host] depends on (via srcs):
  <strong>| genrule //test:src</strong> [host] depends on (via tools):
  <strong>* cc\_binary //test:bin</strong> [host]
Please modify at least one of the dependencies to break the cycle.
</code></pre>

***

Input:

ext.bzl
<pre><code>def foo(name):
      native.genrule(
      name = name,
      outs = ["file.cc"],
      cmd = "touch $@",
)
</code></pre>
BUILD
<pre><code>load(":ext.bzl", "foo")
foo("src2")
files = ["file.cc"]
genrule(
    name = "src",
    outs = var,
    cmd = "touch $@",
)
</code></pre>
Current:
<pre><code>ERROR:/path/BUILD:7:1: generated file 'file.cc' in rule 'src' conflicts with existing generated file from rule 'src2'
</code></pre>
Suggested:
<pre><code>ERROR: /path/BUILD:7:1: <strong>Generated file 'file.cc' in rule 'src' conflicts with existing generated file from rule 'src2'.</strong>
'src' is defined line 7:
    genrule(
        name = "src",
        outs = ["file.cc"],
        cmd = "touch $@",
    )
'src2' is generated by the function foo (line 3) and is equivalent to:
        genrule(
            name = "src2",
            outs = ["file.cc"],
            cmd = "touch $@",
        )
</code></pre>

[Another case](https://github.com/bazelbuild/bazel/issues/1307)

## Suggestions

*   Symbol not found during evaluation
    *   Suggest another symbol from the environment
*   Invalid label (parse error)
    *   Link to label documentation, suggest a fix
*   Directory of the label doesn’t exist
    *   Look in file system and suggest another directory
*   Label name not found
    *   Suggest another name, from the same package
*   Keyword argument doesn’t exist
    *   Suggest name, based on function signature
*   Field not found (obj.nonexistent)
    *   Suggest name, based on list of fields

## Action items

*   Show context line + carret
*   Suggest spelling fixes
*   Show documentation links
*   Show expanded rules (for action conflicts or errors during analysis phase)
*   Improve error messages for rules developers (e.g. pretty-print action
    graph?)
*   Review existing error messages + add ad-hoc suggestions
