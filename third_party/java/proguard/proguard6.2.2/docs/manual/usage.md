To run ProGuard, just type:

`bin/proguard `*options* ...

Typically, you'll put most options in a configuration file (say,
`myconfig.pro`), and just call:

`bin/proguard @myconfig.pro`

You can combine command line options and options from configuration
files. For instance:

`bin/proguard @myconfig.pro -verbose`

You can add comments in a configuration file, starting with a `#`
character and continuing until the end of the line.

Extra whitespace between words and delimiters is ignored. File names
with spaces or special characters should be quoted with single or double
quotes.

Options can be grouped arbitrarily in arguments on the command line and
in lines in configuration files. This means that you can quote arbitrary
sections of command line options, to avoid shell expansion of special
characters, for instance.

The order of the options is generally irrelevant. For quick experiments,
you can abbreviate them to their first unique characters.


## Input/Output Options {: #iooptions}

`@`{: #at} [*filename*](#filename)
: Short for '[`-include`](#include) [*filename*](#filename)'.

`-include`{: #include} [*filename*](#filename)
: Recursively reads configuration options from the given file *filename*.

`-basedirectory`{: #basedirectory} [*directoryname*](#filename)
: Specifies the base directory for all subsequent relative file names in
  these configuration arguments or this configuration file.

`-injars`{: #injars} [*class\_path*](#classpath)
: Specifies the input jars (or apks, aabs, aars, wars, ears, jmods, zips, or
  directories) of the application to be processed. The class files in these
  jars will be processed and written to the output jars. By default, any
  non-class files will be copied without changes. Please be aware of any
  temporary files (e.g. created by IDEs), especially if you are reading your
  input files straight from directories. The entries in the class path can be
  filtered, as explained in the [filters](#filefilters) section. For better
  readability, class path entries can be specified using multiple `-injars`
  options.

`-outjars`{: #outjars} [*class\_path*](#classpath)
: Specifies the names of the output jars (or apks, aabs, aars, wars, ears,
  jmods, zips, or directories). The processed input of the preceding
  [`-injars`](usage.md#injars) options will be written to the named jars. This
  allows you to collect the contents of groups of input jars into
  corresponding groups of output jars. In addition, the output entries can be
  filtered, as explained in the [filters](#filefilters) section. Each
  processed class file or resource file is then written to the first output
  entry with a matching filter, within the group of output jars. You must
  avoid letting the output files overwrite any input files. For better
  readability, class path entries can be specified using multiple
  [`-outjars`](usage.md#outjars) options. Without any
  [`-outjars`](usage.md#outjars) options, no jars will be written.

`-libraryjars`{: #libraryjars} [*class\_path*](#classpath)
: Specifies the library jars (or apks, aabs, aars, wars, ears, jmods, zips,
  directories) of the application to be processed. The files in these jars
  will not be included in the output jars. The specified library jars should
  at least contain the class files that are *extended* by application class
  files. Library class files that are only *called* needn't be present,
  although their presence can improve the results of the optimization step.
  The entries in the class path can be filtered, as explained in the
  [filters](#filefilters) section. For better readability, class path entries
  can be specified using multiple `-libraryjars` options. Please note that the
  boot path and the class path set for running ProGuard are not considered
  when looking for library classes. This means that you explicitly have to
  specify the run-time jar that your code will use. Although this may seem
  cumbersome, it allows you to process applications targeted at different
  run-time environments. For example, you can process [J2SE
  applications](examples.md#application) as well as [JME
  midlets](examples.md#midlet) or [Android
  apps](examples.md#androidapplication), just by specifying the appropriate
  run-time jar.

`-skipnonpubliclibraryclasses`{: #skipnonpubliclibraryclasses}
: Specifies to skip non-public classes while reading library jars, to speed
  up processing and reduce memory usage of ProGuard. By default, ProGuard
  reads non-public and public library classes alike. However, non-public
  classes are often not relevant, if they don't affect the actual program code
  in the input jars. Ignoring them then speeds up ProGuard, without affecting
  the output. Unfortunately, some libraries, including recent JSE run-time
  libraries, contain non-public library classes that are extended by public
  library classes. You then can't use this option. ProGuard will print out
  warnings if it can't find classes due to this option being set.

`-dontskipnonpubliclibraryclasses`{: #dontskipnonpubliclibraryclasses}
: Specifies not to ignore non-public library classes. As of version 4.5,
  this is the default setting.

`-dontskipnonpubliclibraryclassmembers`{: #dontskipnonpubliclibraryclassmembers}
: Specifies not to ignore package visible library class members (fields and
  methods). By default, ProGuard skips these class members while parsing
  library classes, as program classes will generally not refer to them.
  Sometimes however, program classes reside in the same packages as library
  classes, and they do refer to their package visible class members. In those
  cases, it can be useful to actually read the class members, in order to make
  sure the processed code remains consistent.

`-keepdirectories`{: #keepdirectories} \[*[directory\_filter](#filefilters)*\]
: Specifies the directories to be kept in the output jars (or apks, aabs,
  aars, wars, ears, jmods, zips, or directories). By default, directory
  entries are removed. This reduces the jar size, but it may break your
  program if the code tries to find them with constructs like
  "`com.example.MyClass.class.getResource("")`". You'll then want to keep the
  directory corresponding to the package, "`-keepdirectories com.example`". If
  the option is specified without a filter, all directories are kept. With a
  filter, only matching directories are kept. For instance, "`-keepdirectories
  mydirectory`" matches the specified directory, "`-keepdirectories
  mydirectory/*`" matches its immediate subdirectories, and "`-keepdirectories
  mydirectory/**`" matches all of its subdirectories.

`-target`{: #target} *version*
: Specifies the version number to be set in the processed class files. The
  version number can be one of `1.0`,..., `1.9`, or the more recent short
  numbers `5`,..., `12`. By default, the version numbers of the class files
  are left unchanged. For example, you may want to [upgrade class files to
  Java 6](examples.md#upgrade). ProGuard changes their version numbers and
  preverifies them. You can also downgrade class files to older versions than
  Java 8. ProGuard changes their version numbers and backports Java 8
  constructs. ProGuard generally doesn't backport changes in the Java runtime,
  except for the Java 8 stream API and the Java 8 date API, if you add the
  backported libraries `net.sourceforge.streamsupport` and `org.threeten` as
  input, respectively.

`-forceprocessing`{: #forceprocessing}
: Specifies to process the input, even if the output seems up to date. The
  up-to-dateness test is based on a comparison of the date stamps of the
  specified input, output, and configuration files or directories.

## Keep Options {: #keepoptions}

`-keep`{: #keep} \[[,*modifier*](#keepoptionmodifiers),...\] [*class\_specification*](#classspecification)
: Specifies classes and class members (fields and methods) to be preserved
  as entry points to your code. For example, in order to [keep an
  application](examples.md#application), you can specify the main class along
  with its main method. In order to [process a library](examples.md#library),
  you should specify all publicly accessible elements.

`-keepclassmembers`{: #keepclassmembers} \[[,*modifier*](#keepoptionmodifiers),...\] [*class\_specification*](#classspecification)
: Specifies class members to be preserved, if their classes are preserved as
  well. For example, you may want to [keep all serialization fields and
  methods](examples.md#serializable) of classes that implement the
  `Serializable` interface.

`-keepclasseswithmembers`{: #keepclasseswithmembers} \[[,*modifier*](#keepoptionmodifiers),...\] [*class\_specification*](#classspecification)
: Specifies classes and class members to be preserved, on the condition that
  all of the specified class members are present. For example, you may want to
  [keep all applications](examples.md#applications) that have a main method,
  without having to list them explicitly.

`-keepnames`{: #keepnames} [*class\_specification*](#classspecification)
: Short for [`-keep`](#keep),[`allowshrinking`](#allowshrinking)
  [*class\_specification*](#classspecification) Specifies classes and class
  members whose names are to be preserved, if they aren't removed in the
  shrinking phase. For example, you may want to [keep all class
  names](examples.md#serializable) of classes that implement the
  `Serializable` interface, so that the processed code remains compatible with
  any originally serialized classes. Classes that aren't used at all can still
  be removed. Only applicable when obfuscating.

`-keepclassmembernames`{: #keepclassmembernames} [*class\_specification*](#classspecification)
: Short for
  [`-keepclassmembers`](#keepclassmembers),[`allowshrinking`](#allowshrinking)
  [*class\_specification*](#classspecification) Specifies class members whose
  names are to be preserved, if they aren't removed in the shrinking phase.
  For example, you may want to preserve the name of the synthetic `class$`
  methods when [processing a library](examples.md#library) compiled by JDK 1.2
  or older, so obfuscators can detect it again when processing an application
  that uses the processed library (although ProGuard itself doesn't need
  this). Only applicable when obfuscating.

`-keepclasseswithmembernames`{: #keepclasseswithmembernames} [*class\_specification*](#classspecification)
: Short for
  [`-keepclasseswithmembers`](#keepclasseswithmembers),[`allowshrinking`](#allowshrinking)
  [*class\_specification*](#classspecification) Specifies classes and class
  members whose names are to be preserved, on the condition that all of the
  specified class members are present after the shrinking phase. For example,
  you may want to [keep all native method names](examples.md#native) and the
  names of their classes, so that the processed code can still link with the
  native library code. Native methods that aren't used at all can still be
  removed. If a class file is used, but none of its native methods are, its
  name will still be obfuscated. Only applicable when obfuscating.

`-if`{: #if} [*class\_specification*](#classspecification)
: Specifies classes and class members that must be `present` to activate the
  subsequent keep option ([`-keep`](usage.md#keep),
  [`-keepclassmembers`](usage.md#keepclassmembers),...).  The condition and
  the subsequent keep option can share wildcards and references to wildcards.
  For example, you can keep classes on the condition that classes with related
  names exist in your project, with frameworks like
  [Dagger](examples.md#dagger) and [Butterknife](examples.md#butterknife).

`-printseeds`{: #printseeds} \[[*filename*](#filename)\]
: Specifies to exhaustively list classes and class members matched by the
  various `-keep` options. The list is printed to the standard output or to
  the given file. The list can be useful to verify if the intended class
  members are really found, especially if you're using wildcards. For example,
  you may want to list all the [applications](examples.md#applications) or all
  the [applets](examples.md#applets) that you are keeping.

## Shrinking Options {: #shrinkingoptions}

`-dontshrink`{: #dontshrink}
: Specifies not to shrink the input. By default, ProGuard shrinks the code: it
  removes all unused classes and class members. It only keeps the ones listed
  by the various [`-keep`](usage.md#keep) options, and the ones on which they
  depend, directly or indirectly. It also applies a shrinking step after each
  optimization step, since some optimizations may open up the possibility to
  remove more classes and class members.

`-printusage`{: #printusage} \[[*filename*](#filename)\]
: Specifies to list dead code of the input class files. The list is printed
  to the standard output or to the given file. For example, you can [list the
  unused code of an application](examples.md#deadcode). Only applicable when
  shrinking.

`-whyareyoukeeping`{: #whyareyoukeeping} [*class\_specification*](#classspecification)
: Specifies to print details on why the given classes and class members are
  being kept in the shrinking step. This can be useful if you are wondering
  why some given element is present in the output. In general, there can be
  many different reasons. This option prints the shortest chain of methods to
  a specified seed or entry point, for each specified class and class member.
  *In the current implementation, the shortest chain that is printed out may
  sometimes contain circular deductions -- these do not reflect the actual
  shrinking process.* If the [`-verbose`](#verbose) option if specified, the
  traces include full field and method signatures. Only applicable when
  shrinking.

## Optimization Options {: #optimizationoptions}

`-dontoptimize`{: #dontoptimize}
: Specifies not to optimize the input class files. By default, ProGuard
  optimizes all code. It inlines and merges classes and class members, and
  it optimizes all methods at a bytecode level.

`-optimizations`{: #optimizations} [*optimization\_filter*](optimizations.md)
: Specifies the optimizations to be enabled and disabled, at a more
  fine-grained level. Only applicable when optimizing. *This is an expert
  option.*

`-optimizationpasses`{: #optimizationpasses} *n*
: Specifies the number of optimization passes to be performed. By default, a
  single pass is performed. Multiple passes may result in further
  improvements. If no improvements are found after an optimization pass, the
  optimization is ended. Only applicable when optimizing.

`-assumenosideeffects`{: #assumenosideeffects} [*class\_specification*](#classspecification)
: Specifies methods that don't have any side effects, other than possibly
  returning a value. For example, the method `System.currentTimeMillis()`
  returns a value, but it doesn't have any side effects. In the optimization
  step, ProGuard can then remove calls to such methods, if it can determine
  that the return values aren't used. ProGuard will analyze your program code
  to find such methods automatically. It will not analyze library code, for
  which this option can therefore be useful. For example, you could specify
  the method `System.currentTimeMillis()`, so that any idle calls to it will
  be removed. With some care, you can also use the option to [remove logging
  code](examples.md#logging). Note that ProGuard applies the option to the
  entire hierarchy of the specified methods. Only applicable when optimizing.
  In general, making assumptions can be dangerous; you can easily break the
  processed code. *Only use this option if you know what you're doing!*

`-assumenoexternalsideeffects`{: #assumenoexternalsideeffects} [*class\_specification*](#classspecification)
: Specifies methods that don't have any side effects, except possibly on the
  instances on which they are called. This statement is weaker than
  [`-assumenosideeffects`](#assumenosideeffects), because it allows side
  effects on the parameters or the heap. For example, the
  `StringBuffer#append` methods have side effects, but no external side
  effects. This is useful when [removing logging code](examples.md#logging),
  to also remove any related string concatenation code. Only applicable when
  optimizing. Making assumptions can be dangerous; you can easily break the
  processed code. *Only use this option if you know what you're doing!*

`-assumenoescapingparameters`{: #assumenoescapingparameters} [*class\_specification*](#classspecification)
: Specifies methods that don't let their reference parameters escape to the
  heap. Such methods can use, modify, or return the parameters, but not store
  them in any fields, either directly or indirectly. For example, the method
  `System.arrayCopy` does not let its reference parameters escape, but method
  `System.setSecurityManager` does. Only applicable when optimizing. Making
  assumptions can be dangerous; you can easily break the processed code. *Only
  use this option if you know what you're doing!*

`-assumenoexternalreturnvalues`{: #assumenoexternalreturnvalues} [*class\_specification*](#classspecification)
: Specifies methods that don't return reference values that were already on
  the heap when they are called. For example, the `ProcessBuilder#start`
  returns a `Process` reference value, but it is a new instance that wasn't on
  the heap yet. Only applicable when optimizing. Making assumptions can be
  dangerous; you can easily break the processed code. *Only use this option if
  you know what you're doing!*

`-assumevalues`{: #assumevalues} [*class\_specification*](#classspecification)
: Specifies fixed values or ranges of values for primitive fields and
  methods. For example, you can [optimize your app for given Android SDK
  versions](examples.md#androidversions) by specifying the supported range in
  the version constant. ProGuard can then optimize away code paths for older
  versions. Making assumptions can be dangerous; you can easily break the
  processed code. *Only use this option if you know what you're doing!*

`-allowaccessmodification`{: #allowaccessmodification}
: Specifies that the access modifiers of classes and class members may be
  broadened during processing. This can improve the results of the
  optimization step. For instance, when inlining a public getter, it may be
  necessary to make the accessed field public too. Although Java's binary
  compatibility specifications formally do not require this (cfr. [The Java
  Language Specification, Third
  Edition](http://docs.oracle.com/javase/specs/jls/se12/html/index.html),
  [Section
  13.4.6](http://docs.oracle.com/javase/specs/jls/se12/html/jls-13.html#jls-13.4.6)),
  some virtual machines would have problems with the processed code otherwise.
  Only applicable when optimizing (and when obfuscating with the
  [`-repackageclasses`](#repackageclasses) option). *Counter-indication:* you
  probably shouldn't use this option when processing code that is to be used
  as a library, since classes and class members that weren't designed to be
  public in the API may become public.

`-mergeinterfacesaggressively`{: #mergeinterfacesaggressively}
: Specifies that interfaces may be merged, even if their implementing
  classes don't implement all interface methods. This can reduce the size of
  the output by reducing the total number of classes. Note that Java's binary
  compatibility specifications allow such constructs (cfr. [The Java Language
  Specification, Third
  Edition](http://docs.oracle.com/javase/specs/jls/se12/html/index.html),
  [Section
  13.5.3](http://docs.oracle.com/javase/specs/jls/se12/html/jls-13.html#jls-13.5.3)),
  even if they are not allowed in the Java language (cfr. [The Java Language
  Specification, Third
  Edition](http://docs.oracle.com/javase/specs/jls/se12/html/index.html),
  [Section
  8.1.4](http://docs.oracle.com/javase/specs/jls/se12/html/jls-8.html#jls-8.1.4)).
  Only applicable when optimizing.

    *Counter-indication:* setting this option can reduce the performance
    of the processed code on some JVMs, since advanced just-in-time
    compilation tends to favor more interfaces with fewer
    implementing classes. Worse, some JVMs may not be able to handle the
    resulting code. Notably:

    -   Sun's JRE 1.3 may throw an `InternalError` when encountering
        more than 256 *Miranda* methods (interface methods
        without implementations) in a class.

## Obfuscation Options {: #obfuscationoptions}

`-dontobfuscate`{: #dontobfuscate}
: Specifies not to obfuscate the input class files. By default, ProGuard
  obfuscates the code: it assigns new short random names to classes and
  class members. It removes internal attributes that are only useful for
  debugging, such as source files names, variable names, and line numbers.

`-printmapping`{: #printmapping} \[[*filename*](#filename)\]
: Specifies to print the mapping from old names to new names for classes and
  class members that have been renamed. The mapping is printed to the standard
  output or to the given file. For example, it is required for subsequent
  [incremental obfuscation](examples.md#incremental), or if you ever want to
  make sense again of [obfuscated stack traces](examples.md#stacktrace). Only
  applicable when obfuscating.

`-applymapping`{: #applymapping} [*filename*](#filename)
: Specifies to reuse the given name mapping that was printed out in a
  previous obfuscation run of ProGuard. Classes and class members that are
  listed in the mapping file receive the names specified along with them.
  Classes and class members that are not mentioned receive new names. The
  mapping may refer to input classes as well as library classes. This option
  can be useful for [incremental obfuscation](examples.md#incremental), i.e.
  processing add-ons or small patches to an existing piece of code. If the
  structure of the code changes fundamentally, ProGuard may print out warnings
  that applying a mapping is causing conflicts. You may be able to reduce this
  risk by specifying the option
  [`-useuniqueclassmembernames`](#useuniqueclassmembernames) in both
  obfuscation runs. Only a single mapping file is allowed. Only applicable
  when obfuscating.

`-obfuscationdictionary`{: #obfuscationdictionary} [*filename*](#filename)
: Specifies a text file from which all valid words are used as obfuscated
  field and method names. By default, short names like 'a', 'b', etc. are used
  as obfuscated names. With an obfuscation dictionary, you can specify a list
  of reserved key words, or identifiers with foreign characters, for instance.
  White space, punctuation characters, duplicate words, and comments after a
  `#` sign are ignored. Note that an obfuscation dictionary hardly improves
  the obfuscation. Decent compilers can automatically replace them, and the
  effect can fairly simply be undone by obfuscating again with simpler names.
  The most useful application is specifying strings that are typically already
  present in class files (such as 'Code'), thus reducing the class file sizes
  just a little bit more. Only applicable when obfuscating.

`-classobfuscationdictionary`{: #classobfuscationdictionary} [*filename*](#filename)
: Specifies a text file from which all valid words are used as obfuscated
  class names. The obfuscation dictionary is similar to the one of the option
  [`-obfuscationdictionary`](#obfuscationdictionary). Only applicable when
  obfuscating.

`-packageobfuscationdictionary`{: #packageobfuscationdictionary} [*filename*](#filename)
: Specifies a text file from which all valid words are used as obfuscated
  package names. The obfuscation dictionary is similar to the one of the
  option [`-obfuscationdictionary`](#obfuscationdictionary). Only applicable
  when obfuscating.

`-overloadaggressively`{: #overloadaggressively}
: Specifies to apply aggressive overloading while obfuscating. Multiple
  fields and methods can then get the same names, as long as their arguments
  and return types are different, as required by Java bytecode (not just their
  arguments, as required by the Java language). This option can make the
  processed code even smaller (and less comprehensible). Only applicable when
  obfuscating.

    *Counter-indication:* the resulting class files fall within the Java
    bytecode specification (cfr. [The Java Virtual Machine
    Specification](http://docs.oracle.com/javase/specs/jvms/se12/html/index.html),
    first paragraphs of [Section
    4.5](http://docs.oracle.com/javase/specs/jvms/se12/html/jvms-4.html#jvms-4.5)
    and [Section
    4.6](http://docs.oracle.com/javase/specs/jvms/se12/html/jvms-4.html#jvms-4.6)),
    even though this kind of overloading is not allowed in the Java language
    (cfr. [The Java Language Specification, Third
    Edition](http://docs.oracle.com/javase/specs/jls/se12/html/index.html),
    [Section
    8.3](http://docs.oracle.com/javase/specs/jls/se12/html/jls-8.html#jls-8.3)
    and [Section
    8.4.5](http://docs.oracle.com/javase/specs/jls/se12/html/jls-8.html#jls-8.4.5)).
    Still, some tools have problems with it. Notably:

    -   Sun's JDK 1.2.2 `javac` compiler produces an exception when
        compiling with such a library (cfr. [Bug
        \#4216736](http://bugs.sun.com/view_bug.do?bug_id=4216736)). You
        probably shouldn't use this option for processing libraries.
    -   Sun's JRE 1.4 and later fail to serialize objects with
        overloaded primitive fields.
    -   Sun's JRE 1.5 `pack200` tool reportedly has problems with
        overloaded class members.
    -   The class `java.lang.reflect.Proxy` can't handle
        overloaded methods.
    -   Google's Dalvik VM can't handle overloaded static fields.

`-useuniqueclassmembernames`{: #useuniqueclassmembernames}
: Specifies to assign the same obfuscated names to class members that have
  the same names, and different obfuscated names to class members that have
  different names (for each given class member signature). Without the option,
  more class members can be mapped to the same short names like 'a', 'b', etc.
  The option therefore increases the size of the resulting code slightly, but
  it ensures that the saved obfuscation name mapping can always be respected
  in subsequent incremental obfuscation steps.

    For instance, consider two distinct interfaces containing methods with the
    same name and signature. Without this option, these methods may get
    different obfuscated names in a first obfuscation step. If a patch is then
    added containing a class that implements both interfaces, ProGuard will
    have to enforce the same method name for both methods in an incremental
    obfuscation step. The original obfuscated code is changed, in order to
    keep the resulting code consistent. With this option *in the initial
    obfuscation step*, such renaming will never be necessary.

    This option is only applicable when obfuscating. In fact, if you are
    planning on performing incremental obfuscation, you probably want to avoid
    shrinking and optimization altogether, since these steps could remove or
    modify parts of your code that are essential for later additions.

`-dontusemixedcaseclassnames`{: #dontusemixedcaseclassnames}
: Specifies not to generate mixed-case class names while obfuscating. By
  default, obfuscated class names can contain a mix of upper-case characters
  and lower-case characters. This creates perfectly acceptable and usable
  jars. Only if a jar is unpacked on a platform with a case-insensitive filing
  system (say, Windows), the unpacking tool may let similarly named class
  files overwrite each other. Code that self-destructs when it's unpacked!
  Developers who really want to unpack their jars on Windows can use this
  option to switch off this behavior. Obfuscated jars will become slightly
  larger as a result. Only applicable when obfuscating.

`-keeppackagenames`{: #keeppackagenames} \[*[package\_filter](#filters)*\]
: Specifies not to obfuscate the given package names. The optional filter is
  a comma-separated list of package names. Package names can contain **?**,
  **\***, and **\*\*** wildcards, and they can be preceded by the **!**
  negator. Only applicable when obfuscating.

`-flattenpackagehierarchy`{: #flattenpackagehierarchy} \[*package\_name*\]
: Specifies to repackage all packages that are renamed, by moving them into
  the single given parent package. Without argument or with an empty string
  (''), the packages are moved into the root package. This option is one
  example of further [obfuscating package names](examples.md#repackaging). It
  can make the processed code smaller and less comprehensible. Only applicable
  when obfuscating.

`-repackageclasses`{: #repackageclasses} \[*package\_name*\]
: Specifies to repackage all class files that are renamed, by moving them
  into the single given package. Without argument or with an empty string
  (''), the package is removed completely. This option overrides the
  [`-flattenpackagehierarchy`](#flattenpackagehierarchy) option. It is another
  example of further [obfuscating package names](examples.md#repackaging).
  It can make the processed code even smaller and less comprehensible. Its
  deprecated name is `-defaultpackage`. Only applicable when obfuscating.

    *Counter-indication:* classes that look for resource files in their
    package directories will no longer work properly if they are moved
    elsewhere. When in doubt, just leave the packaging untouched by not using
    this option.

    *Note:* On Android, you should not use the empty string when classes like
    activities, views, etc. may be renamed. The Android run-time automatically
    prefixes package-less names in XML files with the application package name
    or with `android.view`. This is unavoidable but it breaks the application
    in this case.

`-keepattributes`{: #keepattributes} \[*[attribute\_filter](attributes.html)*\]
: Specifies any optional attributes to be preserved. The attributes can be
  specified with one or more [`-keepattributes`](usage.md#keepattributes)
  directives. The optional filter is a comma-separated list of [attribute
  names](attributes.html) that Java virtual machines and ProGuard support.
  Attribute names can contain **?**, **\***, and **\*\*** wildcards, and they
  can be preceded by the **!** negator. For example, you should at least keep
  the `Exceptions`, `InnerClasses`, and `Signature` attributes when
  [processing a library](examples.md#library). You should also keep the
  `SourceFile` and `LineNumberTable` attributes for [producing useful
  obfuscated stack traces](examples.md#stacktrace). Finally, you may want to
  [keep annotations](examples.md#annotations) if your code depends on them.
  Only applicable when obfuscating.

`-keepparameternames`{: #keepparameternames}
: Specifies to keep the parameter names and types of methods that are kept.
  This option actually keeps trimmed versions of the debugging attributes
  `LocalVariableTable` and `LocalVariableTypeTable`. It can be useful when
  [processing a library](examples.md#library). Some IDEs can use the
  information to assist developers who use the library, for example with tool
  tips or autocompletion. Only applicable when obfuscating.

`-renamesourcefileattribute`{: #renamesourcefileattribute} \[*string*\]
: Specifies a constant string to be put in the `SourceFile` attributes (and
  `SourceDir` attributes) of the class files. Note that the attribute has to
  be present to start with, so it also has to be preserved explicitly using
  the [`-keepattributes`](usage.md#keepattributes) directive. For example, you
  may want to have your processed libraries and applications produce [useful
  obfuscated stack traces](examples.md#stacktrace). Only applicable when
  obfuscating.

`-adaptclassstrings`{: #adaptclassstrings} \[*[class\_filter](#filters)*\]
: Specifies that string constants that correspond to class names should be
  obfuscated as well. Without a filter, all string constants that correspond
  to class names are adapted. With a filter, only string constants in classes
  that match the filter are adapted. For example, if your code contains a
  large number of hard-coded strings that refer to classes, and you prefer not
  to keep their names, you may want to use this option. Primarily applicable
  when obfuscating, although corresponding classes are automatically kept in
  the shrinking step too.

`-adaptresourcefilenames`{: #adaptresourcefilenames} \[*[file\_filter](#filefilters)*\]
: Specifies the resource files to be renamed, based on the obfuscated names
  of the corresponding class files (if any). Without a filter, all resource
  files that correspond to class files are renamed. With a filter, only
  matching files are renamed. For example, see [processing resource
  files](examples.md#resourcefiles). Only applicable when obfuscating.

`-adaptresourcefilecontents`{: #adaptresourcefilecontents} \[*[file\_filter](#filefilters)*\]
: Specifies the resource files and native libraries whose contents are to be
  updated. Any class names mentioned in the resource files are renamed, based
  on the obfuscated names of the corresponding classes (if any). Any function
  names in the native libraries are renamed, based on the obfuscated names of
  the corresponding native methods (if any). Without a filter, the contents of
  all resource files updated. With a filter, only matching files are updated.
  The resource files are parsed and written using UTF-8 encoding. For an
  example, see [processing resource files](examples.md#resourcefiles). Only
  applicable when obfuscating. *Caveat:* You probably only want to apply this
  option to text files and native libraries, since parsing and adapting
  general binary files as text files can cause unexpected problems. Therefore,
  make sure that you specify a sufficiently narrow filter.

## Preverification Options {: #preverificationoptions}

`-dontpreverify`{: #dontpreverify}
: Specifies not to preverify the processed class files. By default, class
  files are preverified if they are targeted at Java Micro Edition or at Java
  6 or higher. For Java Micro Edition, preverification is required, so you
  will need to run an external preverifier on the processed code if you
  specify this option. For Java 6, preverification is optional, but as of Java
  7, it is required. Only when eventually targeting Android, it is not
  necessary, so you can then switch it off to reduce the processing time a
  bit.

`-microedition`{: #microedition}
: Specifies that the processed class files are targeted at Java Micro
  Edition. The preverifier will then add the appropriate StackMap attributes,
  which are different from the default StackMapTable attributes for Java
  Standard Edition. For example, you will need this option if you are
  [processing midlets](examples.md#midlets).

`-android`{: #android}
: Specifies that the processed class files are targeted at the Android
  platform. ProGuard then makes sure some features are compatible with
  Android. For example, you should specify this option if you are [processing
  an Android application](examples.md#androidapplication).

## General Options {: #generaloptions}

`-verbose`{: #verbose}
: Specifies to write out some more information during processing. If the
  program terminates with an exception, this option will print out the entire
  stack trace, instead of just the exception message.

`-dontnote`{: #dontnote} \[*[class\_filter](#filters)*\]
: Specifies not to print notes about potential mistakes or omissions in the
  configuration, such as typos in class names or missing options that might be
  useful. The optional filter is a regular expression; ProGuard doesn't print
  notes about classes with matching names.

`-dontwarn`{: #dontwarn} \[*[class\_filter](#filters)*\]
: Specifies not to warn about unresolved references and other important
  problems at all. The optional filter is a regular expression; ProGuard
  doesn't print warnings about classes with matching names. Ignoring warnings
  can be dangerous. For instance, if the unresolved classes or class members
  are indeed required for processing, the processed code will not function
  properly. *Only use this option if you know what you're doing!*

`-ignorewarnings`{: #ignorewarnings}
: Specifies to print any warnings about unresolved references and other
  important problems, but to continue processing in any case. Ignoring
  warnings can be dangerous. For instance, if the unresolved classes or class
  members are indeed required for processing, the processed code will not
  function properly. *Only use this option if you know what you're doing!*

`-printconfiguration`{: #printconfiguration} \[[*filename*](#filename)\]
: Specifies to write out the entire configuration that has been parsed, with
  included files and replaced variables. The structure is printed to the
  standard output or to the given file. This can sometimes be useful to
  debug configurations, or to convert XML configurations into a more
  readable format.

`-dump`{: #dump} \[[*filename*](#filename)\]
: Specifies to write out the internal structure of the class files, after
  any processing. The structure is printed to the standard output or to the
  given file. For example, you may want to [write out the contents of a given
  jar file](examples.md#structure), without processing it at all.

`-addconfigurationdebugging`{: #addconfigurationdebugging}
: Specifies to instrument the processed code with debugging statements that
  print out suggestions for missing ProGuard configuration. This can be very
  useful to get practical hints _at run-time_, if your processed code crashes
  because it still lacks some configuration for reflection. For example, the
  code may be [serializing classes with the GSON library](examples.md#gson)
  and you may need some configuration for it. You can generally just
  copy/paste the suggestions from the console into your configuration file.
  *Counter-indication:* do not use this option in release versions, as it adds
  obfuscation information to the processed code.

## Class Paths {: #classpath}

ProGuard accepts a generalization of class paths to specify input files
and output files. A class path consists of entries, separated by the
traditional path separator (e.g. '**:**' on Unix, or '**;**' on Windows
platforms). The order of the entries determines their priorities, in
case of duplicates.

Each input entry can be:

- A class file or resource file,
- An apk file, containing any of the above,
- A jar file, containing any of the above,
- An aar file, containing any of the above,
- A war file, containing any of the above,
- An ear file, containing any of the above,
- A jmod file, containing any of the above,
- A zip file, containing any of the above,
- A directory (structure), containing any of the above.

The paths of directly specified class files and resource files is
ignored, so class files should generally be part of a jar file, an aar
file, a war file, an ear file, a zip file, or a directory. In addition,
the paths of class files should not have any additional directory
prefixes inside the archives or directories.

Each output entry can be:

- An apk file, in which all class files and resource files will
  be collected.
- A jar file, in which any and all of the above will be collected,
- An aar file, in which any and all of the above will be collected,
- A war file, in which any and all of the above will be collected,
- An ear file, in which any and all of the above will be collected,
- A jmod file, in which any and all of the above will be collected,
- A zip file, in which any and all of the above will be collected,
- A directory, in which any and all of the above will be collected.

When writing output entries, ProGuard generally packages the results
in a sensible way, reconstructing the input entries as much as required.
Writing everything to an output directory is the most straightforward
option: the output directory will contain a complete reconstruction of
the input entries. The packaging can be almost arbitrarily complex
though: you could process an entire application, packaged in a zip file
along with its documentation, writing it out as a zip file again. The
Examples section shows a few ways to [restructure output
archives](examples.md#restructuring).

Files and directories can be specified as discussed in the section on
[file names](#filename) below.

In addition, ProGuard provides the possibility to filter the class path
entries and their contents, based on their full relative file names.
Each class path entry can be followed by up to 8 types of [file
filters](#filefilters) between parentheses, separated by semi-colons:

- A filter for all jmod names that are encountered,
- A filter for all aar names that are encountered,
- A filter for all apk names that are encountered,
- A filter for all zip names that are encountered,
- A filter for all ear names that are encountered,
- A filter for all war names that are encountered,
- A filter for all jar names that are encountered,
- A filter for all class file names and resource file names that
  are encountered.

If fewer than 8 filters are specified, they are assumed to be the latter
filters. Any empty filters are ignored. More formally, a filtered class
path entry looks like this:

    classpathentry([[[[[[[jmodfilter;]aarfilter;]apkfilter;]zipfilter;]earfilter;]warfilter;]jarfilter;]filefilter)

Square brackets "\[\]" mean that their contents are optional.

For example, "`rt.jar(java/**.class,javax/**.class)`" matches all class
files in the `java` and `javax` directories inside the `rt` jar.

For example, "`input.jar(!**.gif,images/**)`" matches all files in the
`images` directory inside the `input` jar, except gif files.

The different filters are applied to all corresponding file types,
irrespective of their nesting levels in the input; they are orthogonal.

For example, "`input.war(lib/**.jar,support/**.jar;**.class,**.gif)`"
only considers jar files in the `lib` and `support` directories in the
`input` war, not any other jar files. It then matches all class files
and gif files that are encountered.

The filters allow for an almost infinite number of packaging and
repackaging possibilities. The Examples section provides a few more
examples for [filtering input and output](examples.md#filtering).

## File Names {: #filename}

ProGuard accepts absolute paths and relative paths for the various file
names and directory names. A relative path is interpreted as follows:

- relative to the base directory, if set, or otherwise
- relative to the configuration file in which it is specified, if any,
  or otherwise
- relative to the working directory.

The names can contain Java system properties (or Ant properties, when
using Ant), delimited by angular brackets, '**&lt;**' and '**&gt;**'.
The properties are automatically replaced by their corresponding values.

For example, `<java.home>/lib/rt.jar` is automatically expanded to
something like `/usr/local/java/jdk/jre/lib/rt.jar`. Similarly,
`<user.home>` is expanded to the user's home directory, and `<user.dir>`
is expanded to the current working directory.

Names with special characters like spaces and parentheses must be quoted
with single or double quotes. Each file name in a list of names has to
be quoted individually. Note that the quotes themselves may need to be
escaped when used on the command line, to avoid them being gobbled by
the shell.

For example, on the command line, you could use an option like
`'-injars "my program.jar":"/your directory/your program.jar"'`.

## File Filters {: #filefilters}

Like general [filters](#filters), a file filter is a comma-separated
list of file names that can contain wildcards. Only files with matching
file names are read (in the case of input jars), or written (in the case
of output jars). The following wildcards are supported:

|      |
|------|-----------------------------------------------------------------------------------------
| `?`  | matches any single character in a file name.
| `*`  | matches any part of a filename not containing the directory separator.
| `**` | matches any part of a filename, possibly containing any number of directory separators.

For example, "`java/**.class,javax/**.class`" matches all class files in
the `java` and `javax`.

Furthermore, a file name can be preceded by an exclamation mark '**!**'
to *exclude* the file name from further attempts to match with
*subsequent* file names.

For example, "`!**.gif,images/**`" matches all files in the `images`
directory, except gif files.

The Examples section provides a few more examples for [filtering input
and output](examples.md#filtering).

## Filters {: #filters}

ProGuard offers options with filters for many different aspects of the
configuration: names of files, directories, classes, packages,
attributes, optimizations, etc.

A filter is a list of comma-separated names that can contain wildcards.
Only names that match an item on the list pass the filter. The supported
wildcards depend on the type of names for which the filter is being
used, but the following wildcards are typical:

|      |
|------|-----------------------------------------------------------------------------------------------------------
| `?`  | matches any single character in a name.
| `*`  | matches any part of a name not containing the package separator or directory separator.
| `**` | matches any part of a name, possibly containing any number of package separators or directory separators.

For example, "`foo,*bar`" matches the name `foo` and all names ending
with `bar`.

Furthermore, a name can be preceded by a negating exclamation mark
'**!**' to *exclude* the name from further attempts to match with
*subsequent* names. So, if a name matches an item in the filter, it is
accepted or rejected right away, depending on whether the item has a
negator. If the name doesn't match the item, it is tested against the
next item, and so on. It if doesn't match any items, it is accepted or
rejected, depending on the whether the last item has a negator or not.

For example, "`!foobar,*bar`" matches all names ending with `bar`,
except `foobar`.

## Overview of `Keep` Options {: #keepoverview}

The various [`-keep`](usage.md#keep) options for shrinking and obfuscation may seem a bit
confusing at first, but there's actually a pattern behind them. The
following table summarizes how they are related:

| Keep                                                | From being removed or renamed                        | From being renamed
|-----------------------------------------------------|------------------------------------------------------|--------------------------------------------------------------
| Classes and class members                           | [`-keep`](#keep)                                     | [`-keepnames`](#keepnames)
| Class members only                                  | [`-keepclassmembers`](#keepclassmembers)             | [`-keepclassmembernames`](#keepclassmembernames)
| Classes and class members, if class members present | [`-keepclasseswithmembers`](#keepclasseswithmembers) | [`-keepclasseswithmembernames`](#keepclasseswithmembernames)

Each of these [`-keep`](usage.md#keep) options is of course followed by a
[specification](#classspecification) of the classes and class members
(fields and methods) to which it should be applied.

If you're not sure which option you need, you should probably simply use
`-keep`. It will make sure the specified classes and class members are
not removed in the shrinking step, and not renamed in the obfuscation
step.

!!! warning ""
    -   If you specify a class, without class members, ProGuard only
    preserves the class and its parameterless constructor as
    entry points. It may still remove, optimize, or obfuscate its other
    class members.
    -   If you specify a method, ProGuard only preserves the method as an
    entry point. Its code may still be optimized and adapted.

## Keep Option Modifiers {: #keepoptionmodifiers}

`includedescriptorclasses`
: Specifies that any classes in the type descriptors of the methods and
  fields that the [-keep](#keep) option keeps should be kept as well. This is
  typically useful when [keeping native method names](examples.md#native), to
  make sure that the parameter types of native methods aren't renamed either.
  Their signatures then remain completely unchanged and compatible with the
  native libraries.

`includecode`
: Specifies that code attributes of the methods that the [-keep](#keep)
  option keeps should be kept as well, i.e. may not be optimized or obfuscated.
  This is typically useful for already optimized or obfuscated classes,
  to make sure that their code is not modified during optimization.

`allowshrinking`
: Specifies that the entry points specified in the [-keep](#keep) option may
  be shrunk, even if they have to be preserved otherwise. That is, the entry
  points may be removed in the shrinking step, but if they are necessary after
  all, they may not be optimized or obfuscated.

`allowoptimization`
: Specifies that the entry points specified in the [-keep](#keep) option may
  be optimized, even if they have to be preserved otherwise. That is, the
  entry points may be altered in the optimization step, but they may not be
  removed or obfuscated. This modifier is only useful for achieving unusual
  requirements.

`allowobfuscation`
: Specifies that the entry points specified in the [-keep](#keep) option may
  be obfuscated, even if they have to be preserved otherwise. That is, the
  entry points may be renamed in the obfuscation step, but they may not be
  removed or optimized. This modifier is only useful for achieving unusual
  requirements.

## Class Specifications {: #classspecification}

A class specification is a template of classes and class members (fields
and methods). It is used in the various [`-keep`](usage.md#keep) options and in the
`-assumenosideeffects` option. The corresponding option is only applied
to classes and class members that match the template.

The template was designed to look very Java-like, with some extensions
for wildcards. To get a feel for the syntax, you should probably look at
the [examples](examples.md), but this is an attempt at a complete
formal definition:

    [@annotationtype] [[!]public|final|abstract|@ ...] [!]interface|class|enum classname
        [extends|implements [@annotationtype] classname]
    [{
        [@annotationtype]
        [[!]public|private|protected|static|volatile|transient ...]
        <fields> | (fieldtype fieldname [= values]);

        [@annotationtype]
        [[!]public|private|protected|static|synchronized|native|abstract|strictfp ...]
        <methods> | <init>(argumenttype,...) | classname(argumenttype,...) | (returntype methodname(argumenttype,...) [return values]);
    }]

Square brackets "\[\]" mean that their contents are optional. Ellipsis
dots "..." mean that any number of the preceding items may be specified.
A vertical bar "|" delimits two alternatives. Non-bold parentheses "()"
just group parts of the specification that belong together. The
indentation tries to clarify the intended meaning, but white-space is
irrelevant in actual configuration files.

- The `class` keyword refers to any interface or class. The
  `interface` keyword restricts matches to interface classes. The
  `enum` keyword restricts matches to enumeration classes. Preceding
  the `interface` or `enum` keywords by a `!` restricts matches to
  classes that are not interfaces or enumerations, respectively.

- Every *classname* must be fully qualified, e.g. `java.lang.String`.
  Inner classes are separated by a dollar sign "`$`", e.g.
  `java.lang.Thread$State`. Class names may be specified as regular
  expressions containing the following wildcards:

    |       |
    |-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    | `?`   | matches any single character in a class name, but not the package separator. For example, "`com.example.Test?`" matches "`com.example.Test1`" and "`com.example.Test2`", but not "`com.example.Test12`".
    | `*`   | matches any part of a class name not containing the package separator. For example, "`com.example.*Test*`" matches "`com.example.Test`" and "`com.example.YourTestApplication`", but not "`com.example.mysubpackage.MyTest`". Or, more generally, "`com.example.*`" matches all classes in "`com.example`", but not in its subpackages.
    | `**`  | matches any part of a class name, possibly containing any number of package separators. For example, "`**.Test`" matches all `Test` classes in all packages except the root package. Or, "`com.example.**`" matches all classes in "`com.example`" and in its subpackages.
    | `<n>` | matches the _n_'th matched wildcard in the same option. For example, "`com.example.*Foo<1>`" matches "`com.example.BarFooBar`".

    For additional flexibility, class names can actually be
    comma-separated lists of class names, with optional `!` negators,
    just like file name filters. This notation doesn't look very
    Java-like, so it should be used with moderation.
    For convenience and for backward compatibility, the class name `*`
    refers to any class, irrespective of its package.

- The `extends` and `implements` specifications are typically used to
  restrict classes with wildcards. They are currently equivalent,
  specifying that only classes extending or implementing the given
  class (directly or indirectly) qualify. The given class itself is not
  included in this set. If required, it should be specified in a separate
  option.

- The `@` specifications can be used to restrict classes and class
  members to the ones that are annotated with the specified
  annotation types. An *annotationtype* is specified just like a
  *classname*.

- Fields and methods are specified much like in Java, except that
  method argument lists don't contain argument names (just like in
  other tools like `javadoc` and `javap`). The specifications can also
  contain the following catch-all wildcards:

    |             |
    |-------------|------------------------------
    | `<init>`    | matches any constructor.
    | `<fields>`  | matches any field.
    | `<methods>` | matches any method.
    | `*`         | matches any field or method.

    Note that the above wildcards don't have return types. Only the
    `<init>` wildcard has an argument list.

    Fields and methods may also be specified using regular expressions.
    Names can contain the following wildcards:

    |       |
    |-------|--------------------------------------------------
    | `?`   | matches any single character in a method name.
    | `*`   | matches any part of a method name.
    | `<n>` | matches the _n_'th matched wildcard in the same option.

    Types in descriptors can contain the following wildcards:

    |       |
    |-------|-----------------------------------------------------------------------------------------
    | `%`   | matches any primitive type ("`boolean`", "`int`", etc, but not "`void`").
    | `?`   | matches any single character in a class name.
    | `*`   | matches any part of a class name not containing the package separator.
    | `**`  | matches any part of a class name, possibly containing any number of package separators.
    | `***` | matches any type (primitive or non-primitive, array or non-array).
    | `...` | matches any number of arguments of any type.
    | `<n>` | matches the _n_'th matched wildcard in the same option.

    Note that the `?`, `*`, and `**` wildcards will never match
    primitive types. Furthermore, only the `***` wildcards will match
    array types of any dimension. For example, "`** get*()`" matches
    "`java.lang.Object getObject()`", but not "`float getFloat()`",
    nor "`java.lang.Object[] getObjects()`".

- Constructors can also be specified using their short class names
  (without package) or using their full class names. As in the Java
  language, the constructor specification has an argument list, but no
  return type.

- The class access modifiers and class member access modifiers are
  typically used to restrict wildcarded classes and class members.
  They specify that the corresponding access flags have to be set for
  the member to match. A preceding `!` specifies that the
  corresponding access flag should be unset.

    Combining multiple flags is allowed (e.g. `public static`). It means
    that both access flags have to be set (e.g. `public` *and*
    `static`), except when they are conflicting, in which case at least
    one of them has to be set (e.g. at least `public` *or* `protected`).

    ProGuard supports the additional modifiers `synthetic`, `bridge`,
    and `varargs`, which may be set by compilers.

- With the option [`-assumevalues`](#assumevalues), fields and methods
  with primitive return types can have *values* or *ranges of values*.
  The assignment keyword is `=` or `return`, interchangeably. For
  example, "`boolean flag = true;`" or "`int method() return     5;`".
  Ranges of values are separated by `..`, for example,
  "`int f = 100..200;`". A range includes its begin value and
  end value.
