<!-- THIS HEADER IS FOR input_template_test ONLY -->

Module Docstring: "Input file for input template test"

<a name="#my_example"></a>

## my_example

<pre>
my_example(<a href="#my_example-name">name</a>, <a href="#my_example-useless">useless</a>)
</pre>

Small example of rule using chosen template.

<b>input_template_test BOLD ATTRIBUTES</b>

### Attributes


<b>
      <code>name</code>
        <a href="https://bazel.build/docs/build-ref.html#name">Name</a>; required
</b>
        <p>
          A unique name for this target.
        </p>
<b>
      <code>useless</code>
        String; optional
</b>
        <p>
          This argument will be ignored.
        </p>


<a name="#example"></a>

## example

<pre>
example(<a href="#example-foo">foo</a>, <a href="#example-bar">bar</a>, <a href="#example-baz">baz</a>)
</pre>

Stores information about an example in chosen template.

<b>input_template_test BOLD FIELDS</b>

### Fields

<b>
      <code>foo</code>
</b>
        <p>A string representing foo</p>
<b>
      <code>bar</code>
</b>
        <p>A string representing bar</p>
<b>
      <code>baz</code>
</b>
        <p>A string representing baz</p>


<a name="#my_aspect_impl"></a>

## my_aspect_impl

<pre>
my_aspect_impl(<a href="#my_aspect_impl-ctx">ctx</a>)
</pre>



<b>input_template_test BOLD PARAMETERS</b>

### Parameters

<b>
      <code>ctx</code>
        required.

<a name="#template_function"></a>

## template_function

<pre>
template_function(<a href="#template_function-foo">foo</a>)
</pre>

Runs some checks on the given function parameter.

This rule runs checks on a given function parameter in chosen template.
Use `bazel build` to run the check.


<b>input_template_test BOLD PARAMETERS</b>

### Parameters

<b>
      <code>foo</code>
        required.        <p>
          A unique name for this function.
        </p>


<a name="#my_aspect"></a>

## my_aspect

<pre>
my_aspect(<a href="#my_aspect-name">name</a>, <a href="#my_aspect-first">first</a>)
</pre>

This is my aspect. It does stuff.

### Aspect Attributes

        deps
        String; required.
        attr_aspect
        String; required.

### Attributes

<b>
      <code>name</code>
        <a href="https://bazel.build/docs/build-ref.html#name">Name</a>; required
</b>
        <p>
          A unique name for this target.
        </p>
<b>
      <code>first</code>
        <a href="https://bazel.build/docs/build-ref.html#labels">Label</a>; required
</b>


