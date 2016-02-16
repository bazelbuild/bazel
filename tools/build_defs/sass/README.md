# Sass Rules for Bazel

<div class="toc">
  <h2>Rules</h2>
  <ul>
    <li><a href="#sass_binary">sass_binary</a></li>
    <li><a href="#sass_library">sass_library</a></li>
  </ul>
</div>

## Overview
These build rules are used for building [Sass][sass] projects with Bazel.

[sass]: http://www.sass-lang.com

<a name="setup"></a>
## Setup
To  use the Sass rules, add the following to your `WORKSPACE` file to add the
external repositories for Sass:

```python
load("@bazel_tools//tools/build_defs/sass:sass.bzl", "sass_repositories")

sass_repositories()
```

<a name="basic-example"></a>
## Basic Example

Suppose you have the following directory structure for a simple Sass project:

```
[workspace]/
    WORKSPACE
    hello_world/
        BUILD
        main.scss
    shared/
        BUILD
        _fonts.scss
        _colors.scss
```

`shared/_fonts.scss`

```scss
$default-font-stack: Cambria, "Hoefler Text", Utopia, "Liberation Serif", "Nimbus Roman No9 L Regular", Times, "Times New Roman", ser
if;
$modern-font-stack: Constantia, "Lucida Bright", Lucidabright, "Lucida Serif", Lucida, "DejaVu Serif", "Bitstream Vera Serif", "Liber
ation Serif", Georgia, serif;
```

`shared/_colors.scss`

```scss
$example-blue: #0000ff;
$example-red: #ff0000;
```

`shared/BUILD`

```python
package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/sass:sass.bzl", "sass_library")

sass_library(
    name = "colors",
    srcs = ["_colors.scss"],
)

sass_library(
    name = "fonts",
    srcs = ["_fonts.scss"],
)
```

`hello_world/main.scss`:

```scss
@import "examples/sass/shared/fonts";
@import "examples/sass/shared/colors";

html {
  body {
    font-family: $default-font-stack;
    h1 {
      font-family: $modern-font-stack;
      color: $example-red;
    }
  }
}
```

`hello_world/BUILD:`

```python
package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/sass:sass.bzl", "sass_binary")

sass_binary(
    name = "hello_world",
    src = "main.scss",
    deps = [
         "//shared:colors",
         "//shared:fonts",
    ],
)
```

Build the binary:

```
$ bazel build //hello_world
INFO: Found 1 target...
Target //hello_world:hello_world up-to-date:
  bazel-bin/hello_world/hello_world.css
  bazel-bin/hello_world/hello_world.css.map
INFO: Elapsed time: 1.911s, Critical Path: 0.01s
```

<a name="reference"></a>
## Build Rule Reference

<a name="reference-sass_binary"></a>
## sass_binary

```python
sass_binary(name, src, deps=[], output_style="compressed")
```

Used to generate a CSS artifact from a given `src` sass file.

<table class="table table-condensed table-bordered table-implicit">
  <colgroup>
    <col class="col-param" />
    <col class="param-description" />
  </colgroup>
  <thead>
    <tr>
      <th colspan="2">Implicit Output Targets</th>
    </th>
  </thead>
  <tbody>
    <tr>
      <td><code><strong>name</strong>.css</code></td>
      <td>
        <p>The generated CSS artifact containing all the styles</p>
      </td>
    </tr>
    <tr>
      <td><code><strong>name</strong>.css.map</code></td>
      <td>
        <p>
          <a href="http://thesassway.com/intermediate/using-source-maps-with-sass">source map</a>
          that can be used to optionally debug the generated CSS in a browser.
        </p>
      </td>
    </tr>
  </tbody>
</table>

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
        <p>
          This name will also be used as the name of the generated CSS and source map file of
          this rule.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>src</code></td>
      <td>
        <code>Main source file, required</code>
        <p>The primary Sass source file that will be compiled to CSS.</p>
        <p>
        <code>sass_binary</code> assumes a 1:1 mapping of src to output CSS file (and source map).
        </p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>list of labels, optional</code>
        <p></p>
        <p>
        Each target should be defined using a <code>filegroup</code> rule and should only include "_" prefixed files that are referenced via <code>@import</code> in the target's source file.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>output_style</code></td>
      <td>
        <code>string; optional</code>
        <p>Defaults to <code>compressed</code>.</p>
        <p>
        Can be set to <a href="http://sass-lang.com/documentation/file.SASS_REFERENCE.html#output_style">one of the following</a> output styles defined by <code>sassc</code>.
        </p>
      </td>
    </tr>
  </tbody>
</table>

<a name="reference-sass_library"></a>
## sass_library

```python
sass_library(name, src, deps=[])
```

Used to reference sass a collection of sass files that a
[`sass_binary`](#reference-sass_binary) may depend on (via `@import`
statements), but should not result in any output targets.

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
        <p>
        </p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>a list of labels, required</code>
        <p></p>
        <p>
        <code>sass_library</code> all files should start with an underscore, eg: _colors.scss.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>list of labels, optional</code>
        <p></p>
        <p>
          This could be any other <code>sass_library</code> targets that this target may include.
        </p>
      </td>
    </tr>
  </tbody>
</table>
