**ProGuard** is a Java class file shrinker, optimizer, obfuscator, and
preverifier. The shrinking step detects and removes unused classes,
fields, methods, and attributes. The optimization step analyzes and
optimizes the bytecode of the methods. The obfuscation step renames the
remaining classes, fields, and methods using short meaningless names.
These first steps make the code base smaller, more efficient, and harder
to reverse-engineer. The final preverification step adds preverification
information to the classes, which is required for Java Micro Edition and
for Java 6 and higher.

Each of these steps is optional. For instance, ProGuard can also be used
to just list dead code in an application, or to preverify class files
for efficient use in Java 6.

<div class="center">
  <div class="diagram">
    <div class="row">
      <div class="green box">Input jars</div>
      <div class="right arrow">shrink</div>
      <div class="right arrow">optimize</div>
      <div class="right arrow">obfuscate</div>
      <div class="right arrow">preverify</div>
      <div class="green box">Output jars</div>
    </div>
    <div class="distributed">
      <div class="green box">Library jars</div>
      <div class="right arrow">(unchanged)</div>
      <div class="green box">Library jars</div>
    </div>
  </div>
</div>

ProGuard first reads the **input jars** (or aars, wars, ears, zips, apks, or
directories). It then subsequently shrinks, optimizes, obfuscates, and
preverifies them. You can optionally let ProGuard perform multiple
optimization passes. ProGuard writes the processed results to one or more
**output jars** (or aars, wars, ears, zips, apks, or directories). The input
may contain resource files, whose names and contents can optionally be updated
to reflect the obfuscated class names.

ProGuard requires the **library jars** (or aars, wars, ears, zips, apks, or
directories) of the input jars to be specified. These are essentially the
libraries that you would need for compiling the code. ProGuard uses them to
reconstruct the class dependencies that are necessary for proper processing.
The library jars themselves always remain unchanged. You should still put them
in the class path of your final application.

## Entry points

In order to determine which code has to be preserved and which code can be
discarded or obfuscated, you have to specify one or more *entry points* to
your code. These entry points are typically classes with main methods,
applets, midlets, activities, etc.

- In the **shrinking step**, ProGuard starts from these seeds and recursively
  determines which classes and class members are used. All other classes and
  class members are discarded.
- In the **optimization step**, ProGuard further optimizes the code. Among
  other optimizations, classes and methods that are not entry points can be
  made private, static, or final, unused parameters can be removed, and some
  methods may be inlined.
- In the **obfuscation step**, ProGuard renames classes and class members that
  are not entry points. In this entire process, keeping the entry points
  ensures that they can still be accessed by their original names.
- The **preverification step** is the only step that doesn't have to know the
  entry points.

The [Usage section](usage.md) of this manual describes the necessary [`-keep`
options](usage.md#keepoptions) and the [Examples section](examples.md)
provides plenty of examples.

## Reflection

Reflection and introspection present particular problems for any automatic
processing of code. In ProGuard, classes or class members in your code that
are created or invoked dynamically (that is, by name) have to be specified as
entry points too. For example, `Class.forName()` constructs may refer to any
class at run-time. It is generally impossible to compute which classes have to
be preserved (with their original names), since the class names might be read
from a configuration file, for instance. You therefore have to specify them in
your ProGuard configuration, with the same simple [`-keep`](usage.md#keep)
options.

However, ProGuard already detects and handles the following cases for you:

- `Class.forName("SomeClass")`
- `SomeClass.class`
- `SomeClass.class.getField("someField")`
- `SomeClass.class.getDeclaredField("someField")`
- `SomeClass.class.getMethod("someMethod", null)`
- `SomeClass.class.getMethod("someMethod", new Class[] { A.class,... })`
- `SomeClass.class.getDeclaredMethod("someMethod", null)`
- `SomeClass.class.getDeclaredMethod("someMethod", new Class[] { A.class,... })`
- `AtomicIntegerFieldUpdater.newUpdater(SomeClass.class, "someField")`
- `AtomicLongFieldUpdater.newUpdater(SomeClass.class, "someField")`
- `AtomicReferenceFieldUpdater.newUpdater(SomeClass.class, SomeType.class, "someField")`

The names of the classes and class members may of course be different,
but the constructs should be literally the same for ProGuard to
recognize them. The referenced classes and class members are preserved
in the shrinking phase, and the string arguments are properly updated in
the obfuscation phase.

Furthermore, ProGuard will offer some suggestions if keeping some
classes or class members appears necessary. For example, ProGuard will
note constructs like
"`(SomeClass)Class.forName(variable).newInstance()`". These might be an
indication that the class or interface `SomeClass` and/or its
implementations may need to be preserved. You can then adapt your
configuration accordingly.

Finally, DexGuard can also help for to find less obvious cases of reflection
 _at run-time_. The option
 [`-addconfigurationdebugging`](usage.md#addconfigurationdebugging) lets
 ProGuard instrument the processed code with debugging statements. These print
 out suggestions for missing ProGuard configuration. They can be very useful
 to get practical hints, if your processed code crashes because it still lacks
 some configuration. You can generally just copy/paste the suggestions from
 the console into your configuration file.

For proper results, you should at least be somewhat familiar with the
code that you are processing. Obfuscating code that performs a lot of
reflection may require trial and error, especially without the necessary
information about the internals of the code.
