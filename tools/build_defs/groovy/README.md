# Groovy Rules for Bazel

## Overview

These build rules are used for building [Groovy](http://www.groovy-lang.org/)
projects with Bazel. Groovy libraries may interoperate with and depend on Java
libraries and vice-versa.

* [Setup](#setup)
* [Basic Example](#basic-example)
* [Build Rule Reference](#reference)
  * [`groovy_library`](#groovy_library)
  * [`groovy_and_java_library`](#groovy_and_java_library)
  * [`groovy_binary`](#groovy_binary)

<a name="setup"></a>
## Setup

To be able to use the Groovy rules, you must make the Groovy binaries available
to Bazel. The easiest way to do so is by copying the content of
`groovy.WORKSPACE` to your workspace file and putting `groovy.BUILD` at the root
or your workspace.

<a name="basic-example"></a>
## Basic Example

Suppose you have the following directory structure for a simple Groovy and Java
application:

```
[workspace]/
    WORKSPACE
    app/
        BUILD
        GroovyApp.groovy
    lib/
        BUILD
        GroovyLib.groovy
        JavaLib.java
```

Then, to build the code under lib/, your `lib/BUILD` can look like this:

```python
load("/tools/build_rules/groovy/groovy", "groovy_library")

groovy_library(
    name = "groovylib",
    srcs = glob(["*.groovy"]),
    deps = [
        ":javalib",
    ],
)

java_library(
    name = "javalib",
    srcs = glob(["*.java"]),
)
```

For simplicity, you can combine Groovy and Java sources into a single library
using `groovy_and_java_library`. Note that this allows the Groovy code to
reference the Java code, but not vice-versa. Your `lib/BUILD` file would then
look like this:

```python
load("/tools/build_rules/groovy/groovy", "groovy_and_java_library")

groovy_and_java_library(
    name = "lib",
    srcs = glob(["*.groovy", "*.java"]),
)
```

Finally, you can define a binary using `groovy_binary` as follows:

```python
load("/tools/build_rules/groovy/groovy", "groovy_binary")

groovy_binary(
    name = "GroovyApp",
    srcs = glob(["*.groovy"]),
    main_class = "GroovyApp",
    deps = [
         "//lib",
    ],
)
```

You can then build the application with `bazel build //app:GroovyApp` and run it
with `bazel run //app:GroovyApp`.

<a name="reference"></a>
## Build Rule Reference [reference]

<a name="groovy_library"></a>
### `groovy_library`

`groovy_library(name, srcs, deps, **kwargs)`

<table>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>name</code></td>
      <td>
        <code>Name, required</code>
        <p>A unique name for this rule.</p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>List of .groovy source files used to build the library.</p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels or .jar files, optional</code>
        <p>
          List of other libraries to be included on the compile-time classpath
          when building this library.
        </p>
        <p>
          These can be either other `groovy_library` targets, `java_library`
          targets, `groovy_and_java_library` targets, or raw .jar files.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>**kwargs</code></td>
      <td>
        <code>see <a href="http://bazel.io/docs/build-encyclopedia.html#java_import">java_binary</a></code>
        <p>
          The other arguments of this rule will be passed to the `java_import`
          that wraps the groovy library.
        </p>
      </td>
    </tr>
  </tbody>
</table>

<a name="groovy_and_java_library">
### `groovy_and_java_library`

`groovy_and_java_library(name, srcs, deps, **kwargs)`

<table>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>name</code></td>
      <td>
        <code>Name, required</code>
        <p>A unique name for this rule.</p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>List of .groovy and .java source files used to build the library.</p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels or .jar files, optional</code>
        <p>
          List of other libraries to be included on the compile-time classpath
          when building this library.
        </p>
        <p>
          These can be either other `groovy_library` targets, `java_library`
          targets, `groovy_and_java_library` targets, or raw .jar files.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>**kwargs</code></td>
      <td>
        <code>see <a href="http://bazel.io/docs/build-encyclopedia.html#java_import">java_binary</a></code>
        <p>
          The other arguments of this rule will be passed to the `java_import`
          that wraps the groovy library.
        </p>
      </td>
    </tr>
  </tbody>
</table>

<a name="groovy_binary"></a>
### `groovy_binary`

`groovy_binary(name, main_class, srcs, deps, **kwargs)`

<table>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>name</code></td>
      <td>
        <code>Name, required</code>
        <p>A unique name for this rule.</p>
      </td>
    </tr>
    <tr>
      <td><code>main_class</code></td>
      <td>
        <code>String, required</code>
        <p>
          The name of either a class containing a `main` method or a Groovy
          script file to use as an entry point (see
          <a href="http://www.groovy-lang.org/structure.html#_scripts_versus_classes">
          here</a> for more details on scripts vs. classes).
        </p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>List of .groovy source files used to build the application.</p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels or .jar files, optional</code>
        <p>
          List of other libraries to be included on both the compile-time
          classpath when building this application and the runtime classpath
          when executing it.
        </p>
        <p>
          These can be `groovy_library` targets, `java_library` targets,
          `groovy_and_java_library` targets, or raw .jar files.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>**kwargs</code></td>
      <td>
        <code>see <a href="http://bazel.io/docs/build-encyclopedia.html#java_binary">java_binary</a></code>
        <p>
          The other arguments of this rule will be passed to the `java_binary`
          underlying the `groovy_binary`.
        </p>
      </td>
    </tr>
  </tbody>
</table>
