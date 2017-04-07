---
layout: contribute
title: Parameterized Skylark Aspects
---

#  Design Document: Parameterized Skylark Aspects

**Design documents are not descriptions of the current functionality of Skylark.
Always go to the documentation for current information.**


**Status:** Approved (Proposal #2), Stage 1 implemented.

**Author:** [Dmitry Lomov](mailto:dslomov@google.com),
[Lindley French](mailto:lindleyf@google.com)

**Design document published**: 2016-04-18

*Proposal #2 is  approved*

# Motivation

When rules apply aspects to their dependencies, they often need to
parameterize these aspects with certain values that depend on rule
instances. Typical example:

* `python_proto_library` rule (just like other `*_proto_library` rules)
   need to generate code for different API versions depending on the
   attribute py_api_version in the rule instance


In general, a different set of parameters for aspects means not only
different actions that aspects generate, but also a different set of
extra dependencies that aspects introduce (for example, depending on
the value of py_api_version, python proto aspect will depend on
different versions of python protobuf runtime library).

This functionality is already available for native implementations of
 aspects. Native aspects can be parameterized with
 [AspectParameters](https://github.com/bazelbuild/bazel/blob/72229431c24ad08f0546b03ede9737b633034e30/src/main/java/com/google/devtools/build/lib/packages/AspectParameters.java): (key,value)-dictionaries, where keys and values are simple strings:

1. AspectParameters are produced by *parameter extractor*: a function
 that works on rule instance and produces an aspect parameter dictionary
 based on rule instance attribute values

2. AspectParameters affect both the aspect definitions ([aspect
definition of a particular aspect class depends on AspectParameters](https://github.com/bazelbuild/bazel/blob/f64730fcff20b7d9428e6bd8471ac057ae1bb3b1/src/main/java/com/google/devtools/build/lib/packages/NativeAspectClass.java))
and aspect implementations (AspectParameters are available to [ConfiguredAspectFactory.create](https://github.com/bazelbuild/bazel/blob/0773430188885e075121ebf720c82bb05a39db21/src/main/java/com/google/devtools/build/lib/analysis/ConfiguredAspectFactory.java#L31))

This document describes how to expose this
 functionality to Skylark in safe and powerful way.

# Non-solutions and concerns

*Too much parameterization is bad*. Consider the following strawman:
why cannot we make the entire originating rule instance available to the
propagating aspect? This is very powerful, but it introduces a
_M\*N work problem_: every rule instance originating an aspect will
generate a completely different aspect! Effectively, every rule
originating an aspect will generate an entirely new graph of transitive
dependencies.

In the same vein, it is desirable to always limit the parameter space
across which the aspect parameters might vary.
The good design of Skylark aspect parameterization needs to account for that.

*Using different instances of aspects/rules instead of parameters is
unworkable*. It could be argued that, for example, instead of having
a api_version on python_proto_library, we should have several different
rule classes, py_proto_library_<api version>. This is quite unergonomic.
It is barely bearable for *_proto_library case, and completely
impossible for ndk_cc_library where the potential parameter space is
large (for every parameter combination, a new rule class needs to be
introduced; Skylark macros cannot help here, as Skylark macros cannot
introduce new names into global namespace).

*Increased potential of action conflict.* As it stands now, aspects
output their artifacts to the output directories of the targets they
apply to. This is fragile as unrelated aspects can generate conflicting
actions, and with introduction of parameters the possibility of that
increases (we now have the possibility of the same aspect with different
parameters being applied to a target; aspect author might forget to
disambiguate carefully, leading to subtle and hard to find bugs).


# Solutions

The primary idea for solving the M*N problem is forcing the aspect
author to limit the parameter space and prohibit its accidental
expansion. Instead of having a direct function RI -> AI (where RI is
a rule instance, AI is an aspect instance), we will have (possibly
indirectly) two functions, RI -> P and P -> AI, where P is a finite set
of possible parameter values defined in advance.

## Proposal #1

We introduce the proposal by example (the discussion is below):

```python
SUPPORTED_API_VERSIONS = ["1","2","3"]

def _py_aspect_attrs(api_version):
    if api_version = "1":
        return { '_protoc' : attr.label(default = "//tools/protoc:v1") }
    else if api_version == "2":
….

def _py_aspect_impl(target, ctx, params):
    if params.api_version == "1": ….
py_proto_aspect = aspect(implementation = _py_aspect_impl,
    # params declare all aspect parameters with all possible values
    params = { 'api_version' : set(SUPPORTED_API_VERSIONS) },
    attr_aspects = ['deps'],
    # rhs of attrs can still be dictionary if no dependencies on params
    attrs = _py_aspect_attrs,
)
# Can be omitted, see below.
def _py_proto_library_parameter_extractor(py_api_version, some_other_attr):
    return { 'api_version' : str(py_api_version), }
py_proto_library = rule(implementation = _py_proto_library_impl,
    attrs = {
        'py_api_version' : attr.int()
        'deps': attr.label_list(aspect = py_proto_aspect,
                # Can be omitted: the default extractor
                # just strs all rule attributes with the same
                # names as aspect parameters.
                aspect_parameter_extractor = _py_proto_library_parameter_extractor,
        ),
        'some_other_attr' : attr.string(),
    }
)
```



Here are the elements we introduce:

1. Aspects declare their parameters by means of `params` argument to
 `aspect` function. The value of that argument is a dictionary from
 parameter name to the set of possible values for that parameter.
 We require that the parameter space for an aspect is defined upfront.
 We reject any parameter values that are not declared in advance.
 In this way we address the M*N work problem: we force the aspect
 author to limit the parameter space and prohibit its accidental
 expansion.
 Note: the better expression for this would have been to require params
 to always be of certain enumeration type, but we do not have
 enumeration types in Skylark.

2. We allow aspect attributes (essentially the extra dependencies that
 aspects introduce) to depend on aspect parameters. To this end, we
 allow functions as values of `attrs` argument for `aspect` function.
 If the `attrs` argument is a function it is called with aspect
 parameters to obtain the attributes dictionary (the parameters are
 guaranteed to be within their specified range i.e. set of values).
 If `attrs` argument is a dictionary, it is used as is (compatible
 with current behavior).
 Note: it is possible to extend `attr_aspects` argument in the same way
 as well, if needed.

3. Parameter dictionary is passed as a third parameter to aspect
 implementation function.

4. When rules specify an aspect to apply to their attribute, they can
 optionally specify *a parameter extractor* - a Skylark function that
 produces a parameter dictionary based on values of rule attributes.
 It is an error when a value of parameter produced by a parameter
 extractor is not within its specified range. The default parameter
 extractor just stores the values of rule attributes with the same name
 as parameters of an aspect in question.

### Implementation stages for proposal #1

*Stage 1.* Make the params available to aspect implementation function. This includes:

1. Adding `params` argument to `aspect` function.
 Declared parameters and their ranges become a part of `SkylarkAspect`.

2. Adding appropriate parameter extractor (just the default one,
 str-ing all the relevant attribute values) and introduce the validation
 when creating an aspect in `Attribute.SkylarkRuleAspect`

3. Passing parameter dictionary to aspect implementation function:
 see `SkylarkAspectFactory`.

*Stage 2.* Parameterize Skylark aspect attributes with aspect
parameters. This involves straightforward changes to `aspect` Skylark
function and to `Attribute.SkylarkRuleAspect`. The only tricky thing
there is handling evaluation exceptions from Skylark.

*Stage 3.* Implement custom parameter extractors: a straightforward
change to `Attribute.SkylarkRuleAspect` (most of error handling should be
in place by that stage).

## Proposal #2 (alternative to #1)

In this proposal, aspect parameters are just aspect’s *explicit*
attributes. We restrict the parameter space by requiring all aspect
explicit attributes to have `values` declaration.

Here is how the pervious example will look like in this proposal:

```python
SUPPORTED_API_VERSIONS = ["1","2","3"]

# For rules, configured default function has access to cfg as well, we
# do not support it in aspects
def _py_aspect_protoc(attr_map):
    if attr_map.api_version = "1":
        return Label("//tools/protoc:v1")
      else if attr_map.api_version "2":
        …

def _py_aspect_impl(target, ctx):
    if ctx.attrs.api_version == "1": ….

py_proto_aspect = aspect(implementation = _py_aspect_impl,
    attr_aspects = ['deps'],
    attrs = {
        # For aspect implicit attributes, we allow computed defaults.
        # We still require defaults for all implicit attributes
        '_protoc' : attr.label(default = _py_aspect_protoc)
        # We allow non-implicit attributes. They MUST declare a range of
        # possible values, and they MUST be of a limited set of types
        # (initially just strings)
        'api_version' : attr.string(values = SUPPORTED_API_VERSIONS)
    }
)


# Can be omitted, see below.
def _py_proto_library_parameter_extractor(py_api_version, some_other_attr):
    return { 'api_version' : str(py_api_version), }
py_proto_library = rule(implementation = _py_proto_library_impl,
    attrs = {
        'py_api_version' : attr.int()
        'deps': attr.label_list(aspect = py_proto_aspect,
                # Can be omitted: the default extractor
                # just passes all rule attributes with the same
                # names as aspect non-implicit attributes
                # (aka "parameters").
                aspect_parameter_extractor = _py_proto_library_parameter_extractor,
        ),
        'some_other_attr' : attr.string(),
    }
)
```

Here are the elements we introduce:

1. We limit the types of explicit aspect attributes to "primitive" values
 (strings, ints, booleans).
 Note: initially those attributes should just be strings in line with
 AspectParameters; if we want more types here, we can extend
 AspectParameters to support more types.

2. To facilitate parameterizing aspect dependencies, we allow *implicit*
 aspect attributes to have computed defaults, exposed in the same way
 computed defaults are exposed to Skylark rules: "default value" of
 an attribute can be a function that computes the value given
 an attribute map.
 Note: computed default functions for Skylark rules have access to
 configuration information as well. We cannot support this for aspects
 at the moment; we need to clarify the relationship between aspects and
 configurations, so this is TBD.

3. When rules specify an aspect to apply to their attribute, they can
 optionally specify *a parameter extractor* - a Skylark function that
 produces a parameter dictionary based on values of rule attributes.
 The keys of the computed dictionary must match the names of all
 non-explicit attributes on the aspect. It is an error when a value of
 parameter produced by a parameter extractor is not within its specified
 range. The default parameter extractor just passes values of rule
 attributes with the same name as explicit attributes of an aspect
 in question.

### Implementation stages for proposal #2

(Those stages correspond to implementation stages for proposal #1: at their completion, the same functionality becomes available)

*Stage 1.* Allow explicit attributes with values restriction on aspects:

1. Modify `aspect` value.

2. Add appropriate parameter extractor (just the default one,
 passing through all the relevant attribute values) and introduce the
 validation when creating an aspect in `Attribute.SkylarkRuleAspect`.

3. Ensure that explicit attribute values are passed through to aspect
 implementation function: see `SkylarkAspectFactory`

Stage 1 is [impemented](https://github.com/bazelbuild/bazel/commit/74558fcc8953dec64c2ba5920c8f7a7e3ada36ab).

*Stage 2.* Allow computed defaults for aspect’s implicit attributes.
This involves changes to `aspect` Skylark function and to
`Attribute.SkylarkRuleAspect`. There are two non-obvious parts:

1. we should not allow computed defaults to be default values of
 attributes after AspectDefintion is computed
 (i.e. `SkylarkAspect.getDefinition`)

2. proper error handling is needed here.

*Stage 3.* Implement custom parameter extractors: a straightforward
change to Attribute.SkylarkRuleAspect (most of error handling should
be in place by that stage).

