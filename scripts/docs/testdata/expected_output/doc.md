Project: /_project.yaml
Book: /versions/6.6.6/_book.yaml

# Configurations

{% dynamic setvar version "6.6.6" %}
{% dynamic setvar original_path "/doc" %}
{% include "_buttons.html" %}

A build setting is a single piece of [configuration](/versions/6.6.6/rules/rules#configurations) information.

Like all rules, build setting rules have [implementation functions](https://bazel.build/versions/6.6.6/rules/rules#implementation-function).

In Starlark, transitions are defined much like rules, with a defining
`transition()` [function](lib/transition#transition) and an implementation function.

See [Accessing attributes with transitions](#accessing-attributes-with-transitions)
for how to read these keys.
