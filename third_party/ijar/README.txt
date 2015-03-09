
ijar: A tool for generating interface .jars from normal .jars
=============================================================

Alan Donovan, 26 May 2007.

Rationale:

  In order to improve the speed of compilation of Java programs in
  Bazel, the output of build steps is cached.

  This works very nicely for C++ compilation: a compilation unit
  includes a .cc source file and typically dozens of header files.
  Header files change relatively infrequently, so the need for a
  rebuild is usually driven by a change in the .cc file.  Even after
  syncing a slightly newer version of the tree and doing a rebuild,
  many hits in the cache are still observed.

  In Java, by contrast, a compilation unit involves a set of .java
  source files, plus a set of .jar files containing already-compiled
  JVM .class files.  Class files serve a dual purpose: from the JVM's
  perspective, they are containers of executable code, but from the
  compiler's perspective, they are interface definitions.  The problem
  here is that .jar files are very much more sensitive to change than
  C++ header files, so even a change that is insignificant to the
  compiler (such as the addition of a print statement to a method in a
  prerequisite class) will cause the jar to change, and any code that
  depends on this jar's interface will be recompiled unnecessarily.

  The purpose of ijar is to produce, from a .jar file, a much smaller,
  simpler .jar file containing only the parts that are significant for
  the purposes of compilation.  In other words, an interface .jar
  file.  By changing ones compilation dependencies to be the interface
  jar files, unnecessary recompilation is avoided when upstream
  changes don't affect the interface.

Details:

  ijar is a tool that reads a .jar file and emits a .jar file
  containing only the parts that are relevant to Java compilation.
  For example, it throws away:

  - Files whose name does not end in ".class".
  - All executable method code.
  - All private methods and fields.
  - All constants and attributes except the minimal set necessary to
    describe the class interface.
  - All debugging information
    (LineNumberTable, SourceFile, LocalVariableTables attributes).

  It also sets to zero the file modification times in the index of the
  .jar file.

Implementation:

  ijar is implemented in C++, and runs very quickly.  For example
  (when optimized) it takes only 530ms to process a 42MB
  .jar file containing 5878 classe, resulting in an interface .jar
  file of only 11.4MB in size.  For more usual .jar sizes of a few
  megabytes, a runtime of 50ms is typical.

  The implementation strategy is to mmap both the input jar and the
  newly-created _interface.jar, and to scan through the former and
  emit the latter in a single pass. There are a couple of locations
  where some kind of "backpatching" is required:

  - in the .zip file format, for each file, the size field precedes
    the data.  We emit a zero but note its location, generate and emit
    the stripped classfile, then poke the correct size into the
    location.

  - for JVM .class files, the header (including the constant table)
    precedes the body, but cannot be emitted before it because it's
    not until we emit the body that we know which constants are
    referenced and which are garbage.  So we emit the body into a
    temporary buffer, then emit the header to the output jar, followed
    by the contents of the temp buffer.

  Also note that the zip file format has unnecessary duplication of
  the index metadata: it has header+data for each file, then another
  set of (similar) headers at the end.  Rather than save the metadata
  explicitly in some datastructure, we just record the addresses of
  the already-emitted zip metadata entries in the output file, and
  then read from there as necessary.

Notes:

  This code has no dependency except on the STL and on zlib.

  Almost all of the getX/putX/ReadX/WriteX functions in the code
  advance their first argument pointer, which is passed by reference.

  It's tempting to discard package-private classes and class members.
  However, this would be incorrect because they are a necessary part
  of the package interface, as a Java package is often compiled in
  multiple stages.  For example: in Bazel, both java tests and java
  code inhabit the same Java package but are compiled separately.

Assumptions:

  We assume that jar files are uncompressed v1.0 zip files (created
  with 'jar c0f') with a zero general_purpose_bit_flag.

  We assume that javap/javac don't need the correct CRC checksums in
  the .jar file.

  We assume that it's better simply to abort in the face of unknown
  input than to risk leaving out something important from the output
  (although in the case of annotations, it should be safe to ignore
  ones we don't understand).

TODO:
  Maybe: ensure a canonical sort order is used for every list (jar
  entries, class members, attributes, etc.)  This isn't essential
  because we can assume the compiler is deterministic and the order in
  the source files changes little.  Also, it would require two passes. :(

  Maybe: delete dynamically-allocated memory.

  Add (a lot) more tests.  Include a test of idempotency.
