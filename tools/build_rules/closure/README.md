# Closure Tools for Bazel

## Overview

These rules define targets for JavaScript, stylesheets, and templates that will
be compiled with the Closure Tools toolchain.

* `closure_js_binary`
* `closure_js_library`
* `closure_stylesheet_library`
* `closure_template_library`

## Setup

Add the following to your `WORKSPACE` file to add the external repositories
for the various Closure Tools binaries and the Closure Library:

```python
load("@bazel_tools//tools/build_rules/closure:closure_repositories.bzl", "closure_repositories")

closure_repositories()
```

## Usage

Suppose we are building a web application with template, stylesheet, and
JavaScript files: `hello.soy`, `hello.gss`, and `hello.js`.

`hello.soy`

```
{namespace hello.templates autoescape="strict"}

/**
 * Renders an element containing the text "hello".
 */
{template .hello}
  <div class="{css hello-container}">Hello.</div>
{/template}
```

`hello.gss`

```css
.hello-container {
  color: red;
  font-weight: bold;
}
```

`hello.js`

```javascript
goog.provide('hello');

goog.require('goog.soy');
goog.require('hello.templates');

goog.soy.renderElement(document.body, hello.templates.hello);
```

We can create a BUILD file to compile these and produce two files:

* `hello_combined.css`
* `hello_combined.js`

`BUILD`

```python
load("@bazel_tools//tools/build_rules/closure:closure_js_binary.bzl", "closure_js_binary")
load("@bazel_tools//tools/build_rules/closure:closure_js_library.bzl", "closure_js_library")
load("@bazel_tools//tools/build_rules/closure:closure_stylesheet_library.bzl", "closure_stylesheet_library")
load("@bazel_tools//tools/build_rules/closure:closure_template_library.bzl", "closure_template_library")

closure_js_binary(
    name = "hello",
    main = "hello",
    deps = [":hello_lib"],
)

closure_js_library(
    name = "hello_lib",
    srcs = ["hello.js"],
    deps = [
        "@closure_library//:closure_library",
        "@closure_templates//:closure_templates_js",
        ":hello_css",
        ":hello_soy",
    ]
)

closure_stylesheet_library(
    name = "hello_css",
    srcs = ["hello.gss"],
    deps = ["@closure_library//:closure_library_css"],
)

closure_template_library(
    name = "hello_soy",
    srcs = ["hello.soy"],
)
```

## Known Issues

The version of the Closure Templates compiler that is used will emit warnings
about protected property access and about a missing enum value. These issues
have been fixed in the source, but are not yet available as a downloadable
archive. These warnings are safe to ignore.

You may define a new_local_repository target if you wish to check out the source
from Github yourself. Otherwise you may wish to wait until the Closure tools are
available as Bazel workspaces. e.g.

```python
new_local_repository(
    name = "closure_templates",
    build_file = "tools/build_rules/closure/closure_templates.BUILD",
    path = "/home/user/src/github/google/closure-templates/target",
)
```

