# Java AppEngine Rules for Bazel

## Overview

These build rules are used for building
[Java AppEngine](https://cloud.google.com/appengine/docs/java/)
application with Bazel. It does not aim at general WebApplication
support but can be easily modified to handle a standard WebApplication.

* [Setup](#setup)
* [Basic Example](#basic-example)
* [Build Rule Reference](#reference)
  * [`appengine_war`](#appengine_war)
  * [`java_war`](#java_war)

<a name="setup"></a>
## Setup

To be able to use the Java AppEngine rules, you must make the AppEngine SDK
available to Bazel. The easiest way to do so is by copying the content of
`appengine.WORKSPACE` to your workspace file.

<a name="basic-example"></a>
## Basic Example

Suppose you have the following directory structure for a simple AppEngine
application:

```
[workspace]/
    WORKSPACE
    hello_app/
        BUILD
        java/my/webapp/
            TestServlet.java
        webapp/
            index.html
        webapp/WEB-INF
            web.xml
            appengine-web.xml
```

Then, to build your webapp, your `hello_app/BUILD` can look like:

```python
load("/tools/build_rules/appengine/appengine", "appengine_war")

java_library(
    name = "mylib",
    srcs = ["java/my/webapp/TestServlet.java"],
    deps = [
        "//external:appengine/java/api",
        "//external:javax/servlet/api",
    ],
)

appengine_war(
    name = "myapp",
    jars = [":mylib"],
    data = glob(["webapp/**"]),
    data_path = "webapp",
)
```

For simplicity, you can use the `java_war` rule to build an app from source.
Your `hello_app/BUILD` file would then look like:

```python
load("/tools/build_rules/appengine/appengine", "java_war")

java_war(
    name = "myapp",
    srcs = ["java/my/webapp/TestServlet.java"],
    data = glob(["webapp/**"]),
    data_path = "webapp",
    deps = [
        "//external:appengine/java/api",
        "//external:javax/servlet/api",
    ],
)
```

You can then build the application with `bazel build //hello_app:myapp` and
run in it a development server with `bazel run //hello_app:myapp`. This will
bind a test server on port 8080. If you wish to select another port,
simply append the `--port=12345` to the command-line.

<a name="reference"></a>
## Build Rule Reference [reference]

<a name="appengine_war"></a>
### `appengine_war`

`appengine_war(name, jars, data, data_path)`

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
      <td><code>jars</code></td>
      <td>
        <code>List of labels, required</code>
        <p>
          List of JAR files that will be uncompressed as the code for the
          Web Application.
        </p>
        <p>
          If it is a `java_library` or a `java_import`, the
          JAR from the runtime classpath will be added in the `lib` directory
          of the Web Application.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>data</code></td>
      <td>
        <code>List of files, optional</code>
        <p>List of files used by the Web Application at runtime.</p>
        <p>
          This attribute can be used to specify the list of resources to
          be included into the WAR file.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>data_path</code></td>
      <td>
        <code>String, optional</code>
        <p>Root path of the data.</p>
        <p>
          The directory structure from the data is preserved inside the
          WebApplication but a prefix path determined by `data_path`
          is removed from the the directory structure. This path can
          be absolute from the workspace root if starting with a `/` or
          relative to the rule's directory. It is set to `.` by default.
        </p>
      </td>
    </tr>
  </tbody>
</table>

<a name="java_war">
### `java_war`

`java_war(name, data, data_path, **kwargs)`

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
      <td><code>data</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of files used by the Web Application at runtime.</p>
        <p>Passed to the <a href="#appengine_war">appengine_war</a> rule.</p>
      </td>
    </tr>
    <tr>
      <td><code>data_path</code></td>
      <td>
        <code>String, optional</code>
        <p>Root path of the data.</p>
        <p>Passed to the <a href="#appengine_war">appengine_war</a> rule.</p>
      </td>
    </tr>
    <tr>
      <td><code>**kwargs</code></td>
      <td>
        <code>see <a href="http://bazel.io/docs/build-encyclopedia.html#java_library">java_library</a></code>
        <p>
          The other arguments of this rule will be passed to build a `java_library`
          that will be passed in the `jar` arguments of a
          <a href="#appengine_war">appengine_war</a> rule.
        </p>
      </td>
    </tr>
  </tbody>
</table>
