# Scala Rules for Bazel

<div class="toc">
  <h2>Rules</h2>
  <ul>
    <li><a href="#scala_library">scala_library/scala_macro_library</a></li>
    <li><a href="#scala_binary">scala_binary</a></li>
    <li><a href="#scala_test">scala_test</a></li>
  </ul>
</div>

## Overview

This rule is used for building [Scala][scala] projects with Bazel. There are
currently four rules, `scala_library`, `scala_macro_library`, `scala_binary`
and `scala_test`.

In order to use `scala_library`, `scala_macro_library`, and `scala_binary`,
you must add the following to your WORKSPACE file:
```python
load("@bazel_tools//tools/build_defs/scala:scala.bzl", "scala_repositories")
scala_repositories()
```

[scala]: http://www.scala-lang.org/

<a name="scala_library"></a>
## scala\_library / scala\_macro_library

```python
scala_library(name, srcs, deps, runtime_deps, exports, data, main_class, resources, scalacopts, jvm_flags)
scala_macro_library(name, srcs, deps, runtime_deps, exports, data, main_class, resources, scalacopts, jvm_flags)
```

`scala_library` generates a `.jar` file from `.scala` source files. This rule
also creates an interface jar to avoid recompiling downstream targets unless
then interface changes.

`scala_macro_library` generates a `.jar` file from `.scala` source files when
they contain macros. For macros, there are no interface jars because the macro
code is executed at compile time. For best performance, you want very granular
targets until such time as the zinc incremental compiler can be supported.

In order to make a java rule use this jar file, use the `java_import` rule.

<table class="table table-condensed table-bordered table-params">
  <colgroup>
    <col class="col-param" />
    <col class="param-description" />
  </colgroup>
  <thead>
    <tr>
      <th colspan="2">Attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>name</code></td>
      <td>
        <p><code>Name, required</code></p>
        <p>A unique name for this target</p>
      </td>
    </tr>
      <td><code>srcs</code></td>
      <td>
        <p><code>List of labels, required</code></p>
        <p>List of Scala <code>.scala</code> source files used to build the
        library</p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <p><code>List of labels, optional</code></p>
        <p>List of other libraries to linked to this library target</p>
      </td>
    </tr>
    <tr>
      <td><code>runtime_deps</code></td>
      <td>
        <p><code>List of labels, optional</code></p>
        <p>List of other libraries to put on the classpath only at runtime. This is rarely needed in Scala.</p>
      </td>
    </tr>
    <tr>
      <td><code>exports</code></td>
      <td>
        <p><code>List of labels, optional</code></p>
        <p>List of targets to add to the dependencies of those that depend on this target. Similar
        to the `java_library` parameter of the same name. Use this sparingly as it weakens the
        precision of the build graph.</p>
      </td>
    </tr>
    <tr>
      <td><code>data</code></td>
      <td>
        <p><code>List of labels, optional</code></p>
        <p>List of files needed by this rule at runtime.</p>
      </td>
    </tr>
    <tr>
      <td><code>main_class</code></td>
      <td>
        <p><code>String, optional</code></p>
        <p>Name of class with main() method to use as an entry point</p>
        <p>
          The value of this attribute is a class name, not a source file. The
          class must be available at runtime: it may be compiled by this rule
          (from <code>srcs</code>) or provided by direct or transitive
          dependencies (through <code>deps</code>). If the class is unavailable,
          the binary will fail at runtime; there is no build-time check.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>resources</code></td>
      <td>
        <p><code>List of labels; optional</code></p>
        <p>A list of data files to be included in the JAR.</p>
      </td>
    </tr>
    <tr>
      <td><code>scalacopts</code></td>
      <td>
        <p><code>List of strings; optional</code></p>
        <p>
          Extra compiler options for this library to be passed to scalac. Subject to
          <a href="http://bazel.io/docs/be/make-variables.html">Make variable
          substitution</a> and
          <a href="http://bazel.io/docs/be/common-definitions.html#borne-shell-tokenization">Bourne shell tokenization.</a>
        </p>
      </td>
    </tr>
    <tr>
      <td><code>jvm_flags</code></td>
      <td>
        <p><code>List of strings; optional</code></p>
        <p>
          List of JVM flags to be passed to scalac after the
          <code>scalacopts</code>. Subject to
          <a href="http://bazel.io/docs/be/make-variables.html">Make variable
          substitution</a> and
          <a href="http://bazel.io/docs/be/common-definitions.html#borne-shell-tokenization">Bourne shell tokenization.</a>
        </p>
      </td>
    </tr>
  </tbody>
</table>

<a name="scala_binary"></a>
## scala_binary

```python
scala_binary(name, srcs, deps, runtime_deps, data, main_class, resources, scalacopts, jvm_flags)
```

`scala_binary` generates a Scala executable. It may depend on `scala_library`, `scala_macro_library`
and `java_library` rules.

A `scala_binary` requires a `main_class` attribute.

<table class="table table-condensed table-bordered table-params">
  <colgroup>
    <col class="col-param" />
    <col class="param-description" />
  </colgroup>
  <thead>
    <tr>
      <th colspan="2">Attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>name</code></td>
      <td>
        <p><code>Name, required</code></p>
        <p>A unique name for this target</p>
      </td>
    </tr>
      <td><code>srcs</code></td>
      <td>
        <p><code>List of labels, required</code></p>
        <p>List of Scala <code>.scala</code> source files used to build the
        binary</p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <p><code>List of labels, optional</code></p>
        <p>List of other libraries to linked to this binary target</p>
      </td>
    </tr>
    <tr>
      <td><code>runtime_deps</code></td>
      <td>
        <p><code>List of labels, optional</code></p>
        <p>List of other libraries to put on the classpath only at runtime. This is rarely needed in Scala.</p>
      </td>
    </tr>
    <tr>
      <td><code>data</code></td>
      <td>
        <p><code>List of labels, optional</code></p>
        <p>List of files needed by this rule at runtime.</p>
      </td>
    </tr>
    <tr>
      <td><code>main_class</code></td>
      <td>
        <p><code>String, optional</code></p>
        <p>Name of class with main() method to use as an entry point</p>
        <p>
          The value of this attribute is a class name, not a source file. The
          class must be available at runtime: it may be compiled by this rule
          (from <code>srcs</code>) or provided by direct or transitive
          dependencies (through <code>deps</code>). If the class is unavailable,
          the binary will fail at runtime; there is no build-time check.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>resources</code></td>
      <td>
        <p><code>List of labels; optional</code></p>
        <p>A list of data files to be included in the JAR.</p>
      </td>
    </tr>
    <tr>
      <td><code>scalacopts</code></td>
      <td>
        <p><code>List of strings; optional</code></p>
        <p>
          Extra compiler options for this binary to be passed to scalac. Subject to
          <a href="http://bazel.io/docs/be/make-variables.html">Make variable
          substitution</a> and
          <a href="http://bazel.io/docs/be/common-definitions.html#borne-shell-tokenization">Bourne shell tokenization.</a>
        </p>
      </td>
    </tr>
    <tr>
      <td><code>jvm_flags</code></td>
      <td>
        <p><code>List of strings; optional</code></p>
        <p>
          List of JVM flags to be passed to scalac after the
          <code>scalacopts</code>. Subject to
          <a href="http://bazel.io/docs/be/make-variables.html">Make variable
          substitution</a> and
          <a href="http://bazel.io/docs/be/common-definitions.html#borne-shell-tokenization">Bourne shell tokenization.</a>
        </p>
      </td>
    </tr>
  </tbody>
</table>

<a name="scala_test"></a>
## scala_test

```python
scala_test(name, srcs, suites, deps, data, main_class, resources, scalacopts, jvm_flags)
```

`scala_test` generates a Scala executable which runs unit test suites written
using the `scalatest` library. It may depend on `scala_library`,
`scala_macro_library` and `java_library` rules.

A `scala_test` requires a `suites` attribute, specifying the fully qualified
(canonical) names of the test suites to run. In a future version, we might
investigate lifting this requirement.

