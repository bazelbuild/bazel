A mapping file contains the original names and the obfuscated names of
classes, fields, and methods. ProGuard can write out such a file while
obfuscating an application or a library, with the option
[`-printmapping`](../usage.md#printmapping). ReTrace requires the mapping file
to restore obfuscated stack traces to more readable versions. It is a readable
file with UTF-8 encoding, so you can also look up names in an ordinary text
viewer. The format is pretty self-explanatory, but we describe its details
here.

## Specifications

A mapping file contains a sequence of records of the following form:

    classline
        fieldline *
        methodline *

A `classline`, with a trailing colon, specifies a class and its obfuscated
name:

    originalclassname -> obfuscatedclassname:

A `fieldline`, with 4 leading spaces, specifies a field and its obfuscated
name:

        originalfieldtype originalfieldname -> obfuscatedfieldname

A `methodline`, with 4 leading spaces, specifies a method and its obfuscated
name:

        [startline:endline:]originalreturntype [originalclassname.]originalmethodname(originalargumenttype,...)[:originalstartline[:originalendline]] -> obfuscatedmethodname

An asterisk "*" means the line may occur any number of times. Square brackets
"\[\]" mean that their contents are optional. Ellipsis dots "..." mean that
any number of the preceding items may be specified. The colon ":", the
separator ".", and the arrow "->" are literal tokens.

## Example

The following snippet gives an impression of the structure of a mapping file:

    com example.application.ArgumentWordReader -> com.example.a.a:
        java.lang.String[] arguments -> a
        int index -> a
        36:57:void <init>(java.lang.String[],java.io.File) -> <init>
        64:64:java.lang.String nextLine() -> a
        72:72:java.lang.String lineLocationDescription() -> b
    com.example.application.Main -> com.example.application.Main:
        com.example.application.Configuration configuration -> a
        50:66:void <init>(com.example.application.Configuration) -> <init>
        74:228:void execute() -> a
        2039:2056:void com.example.application.GPL.check():39:56 -> a
        2039:2056:void execute():76 -> a
        2236:2252:void printConfiguration():236:252 -> a
        2236:2252:void execute():80 -> a
        3040:3042:java.io.PrintWriter com.example.application.util.PrintWriterUtil.createPrintWriterOut(java.io.File):40:42 -> a
        3040:3042:void printConfiguration():243 -> a
        3040:3042:void execute():80 -> a
        3260:3268:void readInput():260:268 -> a
        3260:3268:void execute():97 -> a


You can see the names of classes and their fields and methods:

- The fields and methods are listed in ProGuard configuration format (javap
  format), with descriptors that have return types and argument types but no
  argument names. In the above example:

        void <init>(java.lang.String[],java.io.File)

    refers to a constructor with a `String` array argument and a `File`
    argument.

- A method may have a leading line number range, if it is known from the
  original source code (see [Producing useful obfuscated stack
  traces](../examples.md#stacktrace) in the Examples section). Unlike method
  names, line numbers are unique within a class, so ReTrace can resolve lines
  in a stack trace without ambiguities. For example:

        74:228:void execute()

    refers to a method `execute`, defined on lines 74 to 228.

- The obfuscated method name follows the arrow. For example:

        74:228:void execute() -> a

    shows that method `execute` has been renamed to `a`. Multiple fields and
    methods can get the same obfuscated names, as long as their descriptors
    are different.

## Inlined methods

The mapping file accounts for the added complexity of inlined methods (as of
ProGuard/ReTrace version 5.2). The optimization step may inline methods
into other methods &mdash; recursively even. A single line in an obfuscated
stack trace can then correspond to multiple lines in the original stack trace:
the line that throws the exception followed by one or more nested method
calls. In such cases, the mapping file repeats the leading line number range
on subsequent lines. For example:

    3040:3042:java.io.PrintWriter com.example.application.util.PrintWriterUtil.createPrintWriterOut(java.io.File):40:42 -> a
    3040:3042:void printConfiguration():243 -> a
    3040:3042:void execute():80 -> a

- The subsequent lines correspond to the subsequent lines of the original
  stack trace. For example:

        3040:3042:java.io.PrintWriter com.example.application.util.PrintWriterUtil.createPrintWriterOut(java.io.File):40:42 -> a
        3040:3042:void printConfiguration():243 -> a
        3040:3042:void execute():80 -> a

    refers to method `createPrintWriterOut` called from and inlined in
    `printConfiguration`, in turn called from and inlined in method `execute`.

- An original method name may have a preceding class name, if the method
  originates from a different class. For example:

        3040:3042:java.io.PrintWriter com.example.application.util.PrintWriterUtil.createPrintWriterOut(java.io.File):40:42 -> a

    shows that method `createPrintWriterOut` was originally defined in class
    `PrintWriterUtil`.

- A single trailing line number corresponds to an inlined method call. For
  example:

        3040:3042:java.io.PrintWriter com.example.application.util.PrintWriterUtil.createPrintWriterOut(java.io.File):40:42 -> a
        3040:3042:void printConfiguration():243 -> a
        3040:3042:void execute():80 -> a

    specifies that method `execute` called `printConfiguration` on line 80,
    and `printconfiguration` called `createPrintWriterOut` on line 243.

- A traling line number range corresponds to the final inlined method body.
  For example:

        3040:3042:java.io.PrintWriter com.example.application.util.PrintWriterUtil.createPrintWriterOut(java.io.File):40:42 -> a

    shows that method `createPrintWriterOut` covered lines 40 to 42.

- The leading line number range is synthetic, to avoid ambiguities with other
  code in the same class. ProGuard makes up the range, but tries to make it
  similar-looking to the original code (by adding offsets that are multiples
  of 1000), for convenience. For example:

        3040:3042:java.io.PrintWriter com.example.application.util.PrintWriterUtil.createPrintWriterOut(java.io.File):40:42 -> a

    created synthetic range 3040:3042 in the bytecode of class `Main` to be
    unique but still resemble source code range 40:42 in class
    `PrintWriterUtil`.

Tools that don't account for these repeated line number ranges, like older
versions of ReTrace, may still degrade gracefully by outputting the subsequent
lines without interpreting them.
