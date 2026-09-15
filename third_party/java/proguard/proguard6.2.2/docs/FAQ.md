## What is shrinking? {: #shrinking}

Java source code (.java files) is typically compiled to bytecode (.class
files). Bytecode is more compact than Java source code, but it may still
contain a lot of unused code, especially if it includes program libraries.
Shrinking programs such as **ProGuard** can analyze bytecode and remove unused
classes, fields, and methods. The program remains functionally equivalent,
including the information given in exception stack traces.

## What is obfuscation? {: #obfuscation}

By default, compiled bytecode still contains a lot of debugging information:
source file names, line numbers, field names, method names, argument names,
variable names, etc. This information makes it straightforward to decompile
the bytecode and reverse-engineer entire programs. Sometimes, this is not
desirable. Obfuscators such as **ProGuard** can remove the debugging
information and replace all names by meaningless character sequences, making
it much harder to reverse-engineer the code. It further compacts the code as a
bonus. The program remains functionally equivalent, except for the class
names, method names, and line numbers given in exception stack traces.

## What is preverification? {: #preverification}

When loading class files, the class loader performs some sophisticated
verification of the byte code. This analysis makes sure the code can't
accidentally or intentionally break out of the sandbox of the virtual machine.
Java Micro Edition and Java 6 introduced split verification. This means that
the JME preverifier and the Java 6 compiler add preverification information to
the class files (StackMap and StackMapTable attributes, respectively), in
order to simplify the actual verification step for the class loader. Class
files can then be loaded faster and in a more memory-efficient way.
**ProGuard** automatically preverifies the code that it processes.

## What kind of optimizations does ProGuard support? {: #optimization}

Apart from removing unused classes, fields, and methods in the shrinking step,
**ProGuard** can also perform optimizations at the bytecode level, inside and
across methods. Thanks to techniques like control flow analysis, data flow
analysis, partial evaluation, static single assignment, global value
numbering, and liveness analysis, **ProGuard** can:

- Evaluate constant expressions.
- Remove unnecessary field accesses and method calls.
- Remove unnecessary branches.
- Remove unnecessary comparisons and instanceof tests.
- Remove unused code blocks.
- Merge identical code blocks.
- Reduce variable allocation.
- Remove write-only fields and unused method parameters.
- Inline constant fields, method parameters, and return values.
- Inline methods that are short or only called once.
- Simplify tail recursion calls.
- Merge classes and interfaces.
- Make methods private, static, and final when possible.
- Make classes static and final when possible.
- Replace interfaces that have single implementations.
- Perform over 200 peephole optimizations, like replacing `"The answer is
  "+42` by `"The answer is 42"`.
- Optionally remove logging code.

The positive effects of these optimizations will depend on your code and on
the virtual machine on which the code is executed. Simple virtual machines may
benefit more than advanced virtual machines with sophisticated JIT compilers.
At the very least, your bytecode may become a bit smaller.

## Can I use ProGuard to process my commercial application? {: #commercial}

Yes, you can. **ProGuard** itself is distributed under the GPL, but this
doesn't affect the programs that you process. Your code remains yours, and its
license can remain the same.

## Does ProGuard work with Java 2, 5,..., 12? {: #jdk1.4}

Yes, **ProGuard** supports all JDKs from 1.1 up to and including 12. Java 2
introduced some small differences in the class file format. Java 5 added
attributes for generics and for annotations. Java 6 introduced optional
preverification attributes. Java 7 made preverification obligatory and
introduced support for dynamic languages. Java 8 added more attributes and
default methods. Java 9 added support for modules. Java 11 added dynamic
constants and nest-based access control. **ProGuard** handles all versions
correctly.

## Does ProGuard work with Java Micro Edition? {: #jme}

Yes. **ProGuard** itself runs in Java Standard Edition, but you can freely
specify the run-time environment at which your programs are targeted,
including Java Micro Edition. **ProGuard** then also performs the required
preverification, producing more compact results than the traditional external
preverifier.

**ProGuard** also comes with an obfuscator plug-in for the JME Wireless
Toolkit.

## Does ProGuard work for Android apps?

Yes. Google's dx and D8 compilers convert Java bytecode into the Dalvik
bytecode that runs on Android devices. By preprocessing the original bytecode,
**ProGuard** can significantly reduce the file sizes and boost the run-time
performance of the code. It is distributed as part of the Android SDK.
[**DexGuard**](http://www.guardsquare.com/dexguard), **ProGuard**'s
closed-source sibling for Android, offers additional optimizations and more
application protection.

## Does ProGuard have support for Ant? {: #ant}

Yes. **ProGuard** provides an Ant task, so that it integrates seamlessly into
your Ant build process. You can still use configurations in **ProGuard**'s own
readable format. Alternatively, if you prefer XML, you can specify the
equivalent XML configuration.

## Does ProGuard have support for Gradle? {: #gradle}

Yes. **ProGuard** also provides a Gradle task, so that it integrates into your
Gradle build process. You can specify configurations in **ProGuard**'s own
format or embedded in the Groovy configuration.

## Does ProGuard have support for Maven? {: #maven}

**ProGuard**'s jar files are also distributed as artefacts from the [Maven
Central](http://search.maven.org/#search%7Cga%7C1%7Cg:%22net.sf.proguard%22)
repository. There are some third-party plugins that support **ProGuard**, such
as the [android-maven-plugin](http://code.google.com/p/maven-android-plugin/)
and the [IDFC Maven ProGuard Plug-in](http://mavenproguard.sourceforge.net/).
[**DexGuard**](http://www.guardsquare.com/dexguard) also comes with a Maven
plugin.

## Does ProGuard come with a GUI? {: #gui}

Yes. First of all, **ProGuard** is perfectly usable as a command-line tool
that can easily be integrated into any automatic build process. For casual
users, there's also a graphical user interface that simplifies creating,
loading, editing, executing, and saving ProGuard configurations.

## Does ProGuard handle `Class.forName` calls? {: #forname}

Yes. **ProGuard** automatically handles constructs like
`Class.forName("SomeClass")` and `SomeClass.class`. The referenced classes are
preserved in the shrinking phase, and the string arguments are properly
replaced in the obfuscation phase.

With variable string arguments, it's generally not possible to determine their
possible values. They might be read from a configuration file, for instance.
However, **ProGuard** will note a number of constructs like
"`(SomeClass)Class.forName(variable).newInstance()`". These might be an
indication that the class or interface `SomeClass` and/or its implementations
may need to be preserved. The developer can adapt his configuration
accordingly.

## Does ProGuard handle resource files? {: #resource}

Yes. **ProGuard** copies all non-class resource files, optionally adapting
their names and their contents to the obfuscation that has been applied.

## Does ProGuard encrypt string constants? {: #encrypt}

No. String encryption in program code has to be perfectly reversible by
definition, so it only improves the obfuscation level. It increases the
footprint of the code. However, by popular demand, **ProGuard**'s
closed-source sibling for Android,
[**DexGuard**](http://www.guardsquare.com/dexguard), does provide string
encryption, along with more protection techniques against static and dynamic
analysis.

## Does ProGuard perform control flow obfuscation?

No. Control flow obfuscation injects additional branches into the bytecode, in
an attempt to fool decompilers. **ProGuard** does not do this, except to some
extent in its optimization techniques. **ProGuard**'s closed-source sibling
for Android, [**DexGuard**](http://www.guardsquare.com/dexguard), does offer
control flow obfuscation, as one of the many additional techniques to harden
Android apps.

## Does ProGuard support incremental obfuscation? {: #incremental}

Yes. This feature allows you to specify a previous obfuscation mapping file in
a new obfuscation step, in order to produce add-ons or patches for obfuscated
code.

## Can ProGuard obfuscate using reserved keywords? {: #keywords}

Yes. You can specify your own obfuscation dictionary, such as a list of
reserved key words, identifiers with foreign characters, random source files,
or a text by Shakespeare. Note that this hardly improves the obfuscation.
Decent decompilers can automatically replace reserved keywords, and the effect
can be undone fairly easily, by obfuscating again with simpler names.

## Can ProGuard reconstruct obfuscated stack traces? {: #stacktrace}

Yes. **ProGuard** comes with a companion tool, **ReTrace**, that can
'de-obfuscate' stack traces produced by obfuscated applications. The
reconstruction is based on the mapping file that **ProGuard** can write out.
If line numbers have been obfuscated away, a list of alternative method names
is presented for each obfuscated method name that has an ambiguous reverse
mapping. Please refer to the [ProGuard User Manual](manual/index.md) for more
details.

Erik André at Badoo has written a [tool to de-obfuscate HPROF memory
dumps](https://techblog.badoo.com/blog/2014/10/08/deobfuscating-hprof-memory-dumps/).

## How is DexGuard different from ProGuard?

[**DexGuard**](http://www.guardsquare.com/dexguard) is a commercial extension
of **ProGuard**:

- **DexGuard** is specialized for Android applications and libraries: it
  optimizes and obfuscates not just the bytecode, but also the manifest file,
  resources, resource files, asset files, and native libraries.
- **DexGuard** focuses on making apps self-defending against reverse
  engineering and tampering. **DexGuard**'s techniques for obfuscation,
  encryption, and detection are a lot stronger than **ProGuard**'s basic name
  obfuscation.
- **DexGuard** is backward compatible with **ProGuard**: it reads the same
  configuration. It already comes with tuned configuration for the Android
  runtime and for common Android libraries.
