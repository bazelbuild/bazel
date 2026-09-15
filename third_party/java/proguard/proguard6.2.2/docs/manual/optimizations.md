The optimization step of ProGuard can be switched off with the
[`-dontoptimize`](usage.md#dontoptimize) option. For more fine-grained
control over individual optimizations, experts can use the
[`-optimizations`](usage.md#optimizations) option, with a filter based
on the optimization names listed below. The filter works like any
[filter](usage.md#filters) in ProGuard.

The following wildcards are supported:

|     |
|-----|-------------------------------------------------------
| `?` | matches any single character in an optimization name.
| `*` | matches any part of an optimization name.

An optimization that is preceded by an exclamation mark '**!**' is
*excluded* from further attempts to match with *subsequent* optimization
names in the filter. Make sure to specify filters correctly, since they
are not checked for potential typos.

For example,
"**`code/simplification/variable,code/simplification/arithmetic`**" only
performs the two specified peephole optimizations.

For example, "`!method/propagation/*`" performs all optimizations,
except the ones that propagate values between methods.

For example, "`!code/simplification/advanced,code/simplification/*`"
only performs all peephole optimizations.

Some optimizations necessarily imply other optimizations. These are then
indicated. Note that the list is likely to change for newer versions, as
optimizations are added and reorganized.

`library/gson`
: Optimizes usages of the Gson library, whenever possible. See [Gson
  optimization](optimizations.md#gson) for more details.

`class/marking/final`
: Marks classes as final, whenever possible.

`class/unboxing/enum`
: Simplifies enum types to integer constants, whenever possible.

`class/merging/vertical`
: Merges classes vertically in the class hierarchy, whenever possible.

`class/merging/horizontal`
: Merges classes horizontally in the class hierarchy, whenever possible.

`class/merging/wrapper`
: Merges wrapper classes with their wrapped classes, whenever possible.

`field/removal/writeonly`<div>(⇒ `code/removal/advanced`)</div>
: Removes write-only fields.

`field/marking/private`
: Marks fields as private, whenever possible.

`field/propagation/value`<div>(⇒ `code/simplification/advanced`)</div>
: Propagates the values of fields across methods.

`method/marking/private`
: Marks methods as private, whenever possible (*devirtualization*).

`method/marking/static`<div>(⇒ `code/removal/advanced`)</div>
: Marks methods as static, whenever possible (*devirtualization*).

`method/marking/final`
: Marks methods as final, whenever possible.

`method/marking/synchronized`
: Unmarks methods as synchronized, whenever possible.

`method/removal/parameter`<div>(⇒ `code/removal/advanced`)</div>
: Removes unused method parameters.

`method/propagation/parameter`<div>(⇒ `code/simplification/advanced`)</div>
: Propagates the values of method parameters from method invocations to the
  invoked methods.

`method/propagation/returnvalue`<div>(⇒ `code/simplification/advanced`)</div>
: Propagates the values of method return values from methods to their
  invocations.

`method/inlining/short`
: Inlines short methods.

`method/inlining/unique`
: Inlines methods that are only called once.

`method/inlining/tailrecursion`
: Simplifies tail recursion calls, whenever possible.

`code/merging`
: Merges identical blocks of code by modifying branch targets.

`code/simplification/variable`
: Performs peephole optimizations for variable loading and storing.

`code/simplification/arithmetic`
: Performs peephole optimizations for arithmetic instructions.

`code/simplification/cast`
: Performs peephole optimizations for casting operations.

`code/simplification/field`
: Performs peephole optimizations for field loading and storing.

`code/simplification/branch`<div>(⇒ `code/removal/simple`)</div>
: Performs peephole optimizations for branch instructions.

`code/simplification/object`
: Performs peephole optimizations for object instantiation.

`code/simplification/string`
: Performs peephole optimizations for constant strings.

`code/simplification/math`
: Performs peephole optimizations for Math method calls.

`code/simplification/advanced`<div>(*best used with* `code/removal/advanced`)</div>
: Simplifies code based on control flow analysis and data flow analysis.

`code/removal/advanced`<div>(⇒ `code/removal/exception`)</div>
: Removes dead code based on control flow analysis and data flow analysis.

`code/removal/simple`<div>(⇒ `code/removal/exception`)</div>
: Removes dead code based on a simple control flow analysis.

`code/removal/variable`
: Removes unused variables from the local variable frame.

`code/removal/exception`
: Removes exceptions with empty try blocks.

`code/allocation/variable`
: Optimizes variable allocation on the local variable frame.

ProGuard also provides some unofficial settings to control
optimizations, that may disappear in future versions. These are Java
system properties, which can be set as JVM arguments (with `-D...`):

`maximum.inlined.code.length` (default = 8 bytes)
: Specifies the maximum code length (expressed in bytes) of short methods
  that are eligible to be inlined. Inlining methods that are too long may
  unnecessarily inflate the code size.

`maximum.resulting.code.length` (default = 8000 bytes for JSE, 2000 bytes for JME)
: Specifies the maximum resulting code length (expressed in bytes) allowed
  when inlining methods. Many Java virtual machines do not apply just-in-time
  compilation to methods that are too long, so it's important not to let them
  grow too large.

`optimize.conservatively` (default = unset)
: Allows input code with ordinary instructions intentionally throwing
  **`NullPointerException`**, `ArrayIndexOutOfBoundsException`, or
  **`ClassCastException`**, without any other useful purposes. By default,
  ProGuard may just discard such seemingly useless instructions,
  resulting in better optimization of most common code.

## Gson optimization {: #gson}

ProGuard optimizes Gson code by detecting which domain classes are serialized
using the Gson library. It replaces the reflection-based implementation of
GSON for reading and writing fields with injected and optimized code that
accesses the fields of the domain classes directly when reading and writing
JSON. The benefits of this optimization are the following:

- Domain classes used in conjunction with GSON can be freely obfuscated.
- The injected serialization code gives better performance compared to the
  GSON implementation, which relies on reflection.
- Less configuration is needed as the optimization automatically keeps classes
  and fields that are required for serialization.

### Configuration

The Gson optimization is enabled by default and doesn't require any additional
configuration, as long as the application code doesn't use unsupported Gson
features(see [Known limitations](optimizations.md#gsonlimitations)).

### Known limitations {: #gsonlimitations}

ProGuard can not optimize the following use cases of Gson:

- Serializing classes containing one of the following Gson annotations:
    - `@JsonAdapter`
    - `@Since`
    - `@Until`
- Serializing classes that have generic type variables in their signature.
- Serializing classes using a Gson instance that was built with one of the
  following settings on the GsonBuilder:
    - `excludeFieldsWithModifier`
    - `setFieldNamingPolicy`

When one of the above Gson features is used, ProGuard automatically preserves
the original Gson implementation for all affected domain classes.

This means that the serialized fields of these domain classes need to be
explicitely kept again  in the DexGuard configuration so that they can be
safely accessed through reflection.
