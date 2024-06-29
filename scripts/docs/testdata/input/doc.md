Project: /_project.yaml
Book: /_book.yaml

# Configurations

{% include "_buttons.html" %}

A build setting is a single piece of [configuration](/rules/rules#configurations) information.

Like all rules, build setting rules have [implementation functions](https://bazel.build/rules/rules#implementation-function).

In Starlark, transitions are defined much like rules, with a defining
`transition()` [function](lib/transition#transition) and an implementation function.

See [Accessing attributes with transitions](#accessing-attributes-with-transitions)
for how to read these keys.
