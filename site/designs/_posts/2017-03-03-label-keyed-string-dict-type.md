---
layout: contribute
title: Label-keyed String Dictionary Type for Build Attributes
---

# Label-keyed String Dictionary Type for Build Attributes

**Status**: Implemented in
[5e9e194](https://github.com/bazelbuild/bazel/commit/5e9e1949f4c08ce09665b92aadf7ec7e518aab6a)

**Author**: [Michael Staib](mstaib@google.com)

## Background/Motivation

For future work in the realm of allowing Bazel users to define configuration
flags, the `config_setting` rule will need to be able to test configuration
values defined by labels rather than strings. The current solution uses a
dictionary from string (the flag to check the value of) to string (the value
to check against). It makes sense, then, to use a dictionary from label (the
flag to check the value of) to string (the value to check against) for
user-defined configuration which is defined by a label.

Additionally, for work which relates to setting such user-defined configuration,
rules should be able to declare similar dictionaries for the purposes of setting
those same flags.

An example incorporating both testing and setting such flags:

```
flag_rule(
    name = "beep",
    values = ["boop", "bop", "bump"],
    default = "bump"
)

config_setting(
    name = "beep#boop",
    flag_values = {
        ":beep": "boop"
    }
)

transition_rule(
    name = "configuration",
    deps = [
      ":lib"
    ],
    sets_flags = {
      ":beep": "boop"
    }
)

library_rule(
    name = "lib"
    deps = select({
      ":beep#boop": [":boop_dep"],
      "//conditions:default": [":other_dep"]
    })
)
```

## New attribute type: LABEL_KEYED_STRING_DICT
In order to handle these flag values, the BUILD language will need the ability
to express a mapping from a label (a flag's label, to be precise) to a string
(the flag's value). This will be added as `BuildType.LABEL_KEYED_STRING_DICT` in
native rules, and as `attr.label_keyed_string_dict()` in Skylark (taking the
same parameters as `attr.label_list()`). This will have to be serializable
to query `--output=proto` format.

### Native rule representation
Native rules will be able to take the attribute's value using an
`AttributeMapper`, as normal. In this case, the type returned will be
`Map<Label, String>`. In conjunction with `RuleContext.getPrerequisites`, this
can be used to get both the target and the string value associated with it by
iterating over the return value from `getPrerequisites` and looking up the
labels of the `TransitiveInfoCollection`s in the map.

### Skylark representation
Skylark rules must render some representation of this structure in
`ctx.attr.<attrname>`. The only restriction on Skylark dictionary keys is that
they must be immutable, which the various `ConfiguredTarget`s are (although they
must be annotated as such). Accordingly, the value of `ctx.attr.<attrname>` is a
dictionary mapping Target to string. This will have to be changed to be another
special case in the `SkylarkRuleContext`.

Because each target in an attribute will undergo the same transition - if any -
and the transition of the target itself will always be the same, the keys of
this dict will be unique - i.e., there will be no collisions - as long as the
labels used to construct it were unique.

### Handling collisions when converting attribute values from Skylark
Labels are special in that there are multiple ways (and possibly multiple
encodings!) to represent them in a BUILD or Skylark file which are not the same
from Skylark's point of view. In the package `//label`, the strings `"label"`,
`":label"`, `"//label"`, and `"//label:label"` all evaluate to the same `Label`
when they are picked up by Bazel, but they will be different keys in the dict
created by Skylark, where they are merely strings. Skylark does have a label
type (constructed with `Label("//label")`, yet another way of representing the
same label), and Bazel does accept it for `LABEL` attributes, but most uses of
label-type attributes take advantage of Bazel's automatic conversion of strings
in label-type attributes. That conversion does not happen until the Skylark
value enters the build system at a rule attribute, at which point the value may
have been mutated, read, and passed around in Skylark several times.

In Skylark, it is an error for a dictionary literal to contain multiple items
with the same key. For consistency and simplicity, `LABEL_KEYED_STRING_DICT`
will throw a `ConversionException` in its convert method if two of the Skylark
dict's keys evaluate to the same label, even if they also have the same value.
This only covers the case where the two keys are distinct strings; if two
identical keys are used in a dictionary literal, there will be an error in
Skylark before this logic ever sees it. Mutations of a key (i.e.,
`dictionary[key] = value` for a `key` which is already in the dictionary) will
continue to be allowed as normal.

## Testing Plan

* Conversion exception for non-dict values
* Conversion exception for dicts other than string-to-string
* Conversion exception for dicts with multiple keys evaluating to the same label
* Conversion exception for dicts with invalid labels as keys
* Successfully converts to `Map<Label, String>`
* Successfully converts to query proto
* Successfully converts to query XML
* Successfully outputs in build format from query
* visitLabels visits the labels in the keys
* Skylark can define `label_keyed_string_dict` attributes and receive them as a
  dict of Target to string
* Skylark can define `label_keyed_string_dict` attributes with provider
  requirements and have them be respected
* Skylark can define `label_keyed_string_dict` attributes with filetype
  requirements and have them be respected
* Skylark can define `label_keyed_string_dict` attributes and require they not
  be empty
* Skylark can define `label_keyed_string_dict` attributes and make them
  mandatory
* Skylark can define `label_keyed_string_dict` attributes and set the default
  value
* Skylark can define `label_keyed_string_dict` attributes and have Aspects
  follow them
