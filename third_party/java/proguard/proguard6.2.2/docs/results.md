**ProGuard** successfully processes any Java bytecode, ranging from small
Android applications to entire run-time libraries. It primarily reduces the
size of the processed code, with some potential increase in efficiency as an
added bonus. The improvements obviously depend on the original code. The table
below presents some typical results:

|                                                                                                   Input Program | Original size | After shrinking | After optim. | After obfusc. | Total reduction |  Time  | Memory usage
|-----------------------------------------------------------------------------------------------------------------|---------------|-----------------|--------------|---------------|-----------------|--------|--------------
| [Worm](http://www.oracle.com/technetwork/java/javame/index.html), a sample midlet from Oracle's JME             |        10.3 K |           9.8 K |        9.6 K |         8.5 K |            18 % |    2 s |         19 M
| [Javadocking](http://www.javadocking.com/), a docking library                                                   |         290 K |           281 K |        270 K |         201 K |            30 % |   12 s |         32 M
| **ProGuard** itself                                                                                             |         648 K |           579 K |        557 K |         348 K |            46 % |   28 s |         66 M
| [JDepend](http://www.clarkware.com/software/JDepend.html), a Java quality metrics tool                          |          57 K |            36 K |         33 K |          28 K |            51 % |    6 s |         24 M
| [the run-time classes](http://www.oracle.com/technetwork/java/javase/overview/index.html) from Oracle's Java 6 |          53 M |            23 M |         22 M |          18 M |            66 % | 16 min |        270 M
| [Tomcat](http://tomcat.apache.org/), the Apache servlet container                                               |         1.1 M |           466 K |        426 K |         295 K |            74 % |   17 s |         44 M
| [JavaNCSS](http://javancss.codehaus.org/), a Java source metrics tool                                           |         632 K |           242 K |        212 K |         152 K |            75 % |   20 s |         36 M
| [Ant](http://ant.apache.org/), the Apache build tool                                                            |         2.4 M |           401 K |        325 K |         242 K |            90 % |   23 s |         61 M

Results were measured with ProGuard 4.0 on a 2.6 GHz Pentium 4 with 512 MB of
memory, using Sun JDK 1.5.0 in Fedora Core 3 Linux. All of this technology and
software has evolved since, but the gist of the results remains the same.

The program sizes include companion libraries. The shrinking step produces the
best results for programs that use only small parts of their libraries. The
obfuscation step can significantly shrink large programs even further, since
the identifiers of their many internal references can be replaced by short
identifiers.

The Java 6 run-time classes are the most complex example. The classes perform
a lot of introspection, interacting with the native code of the virtual
machine. The 1500+ lines of configuration were largely composed by automated
analysis, complemented by a great deal of trial and error. The configuration
is probably not complete, but the resulting library successfully serves as a
run-time environment for running applications like ProGuard and the ProGuard
GUI.

For small inputs, timings are governed by the reading and parsing of the jars.
For large inputs, the optimization step becomes more important. For instance,
processing the Java 6 run-time classes without optimization only takes 2
minutes.

Memory usage (the amount of physical memory used by ProGuard while processing)
is governed by the basic java virtual machine and by the total size of the
library jars and program jars.
