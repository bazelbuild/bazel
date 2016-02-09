# Groovy Rules

<div class="toc">
  <h2>Rules</h2>
  <ul>
    <li><a href="#groovy_library">groovy_library</a></li>
    <li><a href="#groovy_and_java_library">groovy_and_java_library</a></li>
    <li><a href="#groovy_binary">groovy_binary</a></li>
    <li><a href="#groovy_junit_test">groovy_junit_test</a></li>
    <li><a href="#spock_test">spock_test</a></li>
  </ul>
</div>

## Overview

These build rules are used for building [Groovy](http://www.groovy-lang.org/)
projects with Bazel. Groovy libraries may interoperate with and depend on Java
libraries and vice-versa.

<a name="setup"></a>
## Setup

To be able to use the Groovy rules, you must provide bindings for the following
targets:

  * `//external:groovy-sdk`, pointing at the
    [Groovy SDK binaries](http://www.groovy-lang.org/download.html)
  * `//external:groovy`, pointing at the Groovy core language jar
  * `//external:junit`, pointing at JUnit (only required if using the test rules)
  * `//external:spock`, pointing at Spock (only required if using `spock_test`)

The easiest way to do so is to add the following to your `WORKSPACE` file and
putting `groovy.BUILD` at the root of your workspace:

```python
load("@bazel_tools//tools/build_defs/groovy:groovy.bzl", "groovy_repositories")

groovy_repositories()
```

<a name="basic-example"></a>
## Basic Example

Suppose you have the following directory structure for a simple Groovy and Java
application:

```
[workspace]/
    WORKSPACE
    src/main/groovy/
        app/
            BUILD
            GroovyApp.groovy
        lib/
            BUILD
            GroovyLib.groovy
            JavaLib.java
    src/test/groovy/
        lib/
            BUILD
            LibTest.groovy
```

Then, to build the code under src/main/groovy/lib/, your
`src/main/groovy/lib/BUILD` can look like this:

```python
load("@bazel_tools//tools/build_defs/groovy:groovy.bzl", "groovy_library")

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
reference the Java code, but not vice-versa. Your `src/main/groovy/lib/BUILD`
file would then look like this:

```python
load("@bazel_tools//tools/build_defs/groovy:groovy.bzl", "groovy_and_java_library")

groovy_and_java_library(
    name = "lib",
    srcs = glob(["*.groovy", "*.java"]),
)
```

To build the application under src/main/groovy/app, you can define a binary
using `groovy_binary` as follows:

```python
load("@bazel_tools//tools/build_defs/groovy:groovy.bzl", "groovy_binary")

groovy_binary(
    name = "GroovyApp",
    srcs = glob(["*.groovy"]),
    main_class = "GroovyApp",
    deps = [
         "//src/main/groovy/lib",
    ],
)
```

Finally, you can write tests in Groovy using `groovy_test`. The `srcs` of this
rule will be converted into names of class files that are passed to JUnit. For
this to work, the test sources must be under src/test/groovy or src/test/java.
To build the test under src/test/groovy/lib, your BUILD file would look like
this:

```python
load("@bazel_tools//tools/build_defs/groovy:groovy.bzl", "groovy_test", "groovy_library")


groovy_library(
  name = "testlib",
  srcs = glob(["*.groovy"]),
)

groovy_test(
  name = "LibTest",
  srcs = ["LibTest.groovy"],
  deps = [":testlib"],
)
```

If you're using JUnit or Spock, see
<a href="#groovy_junit_test">groovy_junit_test</a> or
<a href="#spock_test>spock_test</a> for wrappers that make testing with these
systems slightly more convenient.

<a name="groovy_library"></a>
## groovy_library

```python
groovy_library(name, srcs, deps, **kwargs)
```

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
        <code>see <a href="http://bazel.io/docs/be/java.html#java_import">java_binary</a></code>
        <p>
          The other arguments of this rule will be passed to the `java_import`
          that wraps the groovy library.
        </p>
      </td>
    </tr>
  </tbody>
</table>

<a name="groovy_and_java_library"></a>
## groovy\_and\_java\_library

```python
groovy_and_java_library(name, srcs, deps, **kwargs)
```

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
        <code>see <a href="http://bazel.io/docs/be/java.html#java_import">java_binary</a></code>
        <p>
          The other arguments of this rule will be passed to the `java_import`
          that wraps the groovy library.
        </p>
      </td>
    </tr>
  </tbody>
</table>

<a name="groovy_binary"></a>
## groovy_binary

```python
groovy_binary(name, main_class, srcs, deps, **kwargs)
```

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
        <code>see <a href="http://bazel.io/docs/be/java.html#java_binary">java_binary</a></code>
        <p>
          The other arguments of this rule will be passed to the `java_binary`
          underlying the `groovy_binary`.
        </p>
      </td>
    </tr>
  </tbody>
</table>

<a name="groovy_test"></a>
## groovy_test

```python
groovy_test(name, deps, srcs, data, resources, jvm_flags, size, tags)
```

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
        <code>Name, required</code>
        <p>A unique name for this rule.</p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>
          List of .groovy source files whose names will be converted to classes
          passed to JUnitCore.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels or .jar files, optional</code>
        <p>
          List of libraries to be included on both the compile-time classpath
          when building this test and on the runtime classpath when executing it.
        </p>
        <p>
          These can be `groovy_library` targets, `java_library` targets,
          `groovy_and_java_library` targets, or raw .jar files.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>resources</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>
          A list of data files to include on the test's classpath. This is
          accomplished by creating a `java_library` containing only the specified
          resources and including that library in the test's dependencies.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>jvm_flags</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>
          A list of flags to embed in the wrapper script generated for running
          this binary.
        </p>
      </td>
    </tr>
  </tbody>
</table>

<a name="groovy_junit_test"></a>
## groovy_junit_test

```python
groovy_junit_test(name, tests, deps, groovy_srcs, java_srcs, data, resources, jvm_flags, size, tags)
```

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
        <code>Name, required</code>
        <p>A unique name for this rule.</p>
      </td>
    </tr>
    <tr>
      <td><code>tests</code></td>
      <td>
        <code>List of labels, required</code>
        <p>
          List of .groovy source files that will be used as test specifications
          that will be executed by JUnit.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>groovy_srcs</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>
          List of additional .groovy source files that will be used to build the
          test.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>java_srcs</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>
          List of additional .java source files that will be used to build the
          test.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels or .jar files, optional</code>
        <p>
          List of libraries to be included on both the compile-time classpath
          when building this test and on the runtime classpath when executing it.
        </p>
        <p>
          These can be `groovy_library` targets, `java_library` targets,
          `groovy_and_java_library` targets, or raw .jar files.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>resources</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>
          A list of data files to include on the test's classpath. This is
          accomplished by creating a `java_library` containing only the specified
          resources and including that library in the test's dependencies.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>jvm_flags</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>
          A list of flags to embed in the wrapper script generated for running
          this binary.
        </p>
      </td>
    </tr>
  </tbody>
</table>

<a name="spock_test"></a>
## spock_test

```python
spock_test(name, specs, deps, groovy_srcs, java_srcs, data, resources, jvm_flags, size, tags)
```

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
        <code>Name, required</code>
        <p>A unique name for this rule.</p>
      </td>
    </tr>
    <tr>
      <td><code>specs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>
          List of .groovy source files that will be used as test specifications
          that will be executed by JUnit.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>groovy_srcs</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>
          List of additional .groovy source files that will be used to build the
          test.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>java_srcs</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>
          List of additional .java source files that will be used to build the
          test.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels or .jar files, optional</code>
        <p>
          List of libraries to be included on both the compile-time classpath
          when building this test and on the runtime classpath when executing it.
        </p>
        <p>
          These can be `groovy_library` targets, `java_library` targets,
          `groovy_and_java_library` targets, or raw .jar files.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>resources</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>
          A list of data files to include on the test's classpath. This is
          accomplished by creating a `java_library` containing only the specified
          resources and including that library in the test's dependencies.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>jvm_flags</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>
          A list of flags to embed in the wrapper script generated for running
          this binary.
        </p>
      </td>
    </tr>
  </tbody>
</table>
