While preparing a configuration for processing your code, you may bump
into a few problems. The following sections discuss some common issues
and solutions:


## Problems while processing {: #processing}

ProGuard may print out some notes and non-fatal warnings:

**Note: can't find dynamically referenced class ...** {: #dynamicalclass}
: ProGuard can't find a class or interface that your code is accessing by
  means of introspection. You should consider adding the jar that contains
  this class.

**Note: ... calls '(...)Class.forName(variable).newInstance()'** {: #dynamicalclasscast}
: Your code uses reflection to dynamically create class instances, with a
  construct like "`(MyClass)Class.forName(variable).newInstance()`". Depending
  on your application, you may need to keep the mentioned classes with an
  option like "`-keep class MyClass`", or their implementations with an option
  like "`-keep class * implements MyClass`". You can switch off these notes by
  specifying the [`-dontnote`](usage.md#dontnote) option.

**Note: ... accesses a field/method '...' dynamically** {: #dynamicalclassmember}
: Your code uses reflection to find a fields or a method, with a construct
  like "`.getField("myField")`". Depending on your application, you may need
  to figure out where the mentioned class members are defined and keep them
  with an option like "`-keep class MyClass { MyFieldType myField; }`".
  Otherwise, ProGuard might remove or obfuscate the class members, since it
  can't know which ones they are exactly. It does list possible candidates,
  for your information. You can switch off these notes by specifying the
  [`-dontnote`](usage.md#dontnote) option.

**Note: ... calls 'Class.get...'**, **'Field.get...'**, or **'Method.get...'** {: #attributes}
: Your code uses reflection to access metadata from the code, with an
  invocation like "`class.getAnnotations()`". You then generally need to
  preserve optional [class file attributes](attributes.md), which ProGuard
  removes by default. The attributes contain information about annotations,
  enclosing classes, enclosing methods, etc. In a summary in the log, ProGuard
  provides a suggested configuration, like [`-keepattributes
  *Annotation*`](usage.md#keepattributes). If you're sure the attributes are
  not necessary, you can switch off these notes by specifying the
  [`-dontnote`](usage.md#dontnote) option.

**Note: the configuration refers to the unknown class '...'** {: #unknownclass}
: Your configuration refers to the name of a class that is not present in
  the program jars or library jars. You should check whether the name is
  correct. Notably, you should make sure that you always specify
  fully-qualified names, not forgetting the package names.

**Note: the configuration keeps the entry point '...', but not the descriptor class '...'** {: #descriptorclass}
: Your configuration contains a [`-keep`](usage.md#keep) option to preserve
  the given method (or field), but no `-keep` option for the given class that
  is an argument type or return type in the method's descriptor. You may then
  want to keep the class too. Otherwise, ProGuard will obfuscate its name,
  thus changing the method's signature. The method might then become
  unfindable as an entry point, e.g. if it is part of a public API. You can
  automatically keep such descriptor classes with the `-keep` option modifier
  [`includedescriptorclasses`](usage.md#includedescriptorclasses)
  (`-keep,includedescriptorclasses` ...). You can switch off these notes by
  specifying the [`-dontnote`](usage.md#dontnote) option.

**Note: the configuration explicitly specifies '...' to keep library class '...'** {: #libraryclass}
: Your configuration contains a [`-keep`](usage.md#keep) option to preserve
  the given library class. However, you don't need to keep any library
  classes. ProGuard always leaves underlying libraries unchanged. You can
  switch off these notes by specifying the [`-dontnote`](usage.md#dontnote)
  option.

**Note: the configuration doesn't specify which class members to keep for class '...'** {: #classmembers}
: Your configuration contains a
  [`-keepclassmembers`](usage.md#keepclassmembers)/[`-keepclasseswithmembers`](usage.md#keepclasseswithmembers)
  option to preserve fields or methods in the given class, but it doesn't
  specify which fields or methods. This way, the option simply won't have any
  effect. You probably want to specify one or more fields or methods, as usual
  between curly braces. You can specify all fields or methods with a wildcard
  "`*;`". You should also consider if you just need the more common
  [`-keep`](usage.md#keep) option, which preserves all specified classes *and*
  class members. The [overview of all `keep` options](usage.md#keepoverview)
  can help. You can switch off these notes by specifying the
  [`-dontnote`](usage.md#dontnote) option.

**Note: the configuration specifies that none of the methods of class '...' have any side effects** {: #nosideeffects}
: Your configuration contains an option
  [`-assumenosideeffects`](usage.md#assumenosideeffects) to indicate that the
  specified methods don't have any side effects. However, the configuration
  tries to match *all* methods, by using a wildcard like "`*;`". This includes
  methods from `java.lang.Object`, such as `wait()` and `notify()`. Removing
  invocations of those methods will most likely break your application. You
  should list the methods without side effects more conservatively. You can
  switch off these notes by specifying the [`-dontnote`](usage.md#dontnote)
  option.

**Note: duplicate definition of program/library class** {: #duplicateclass}
: Your program jars or library jars contain multiple definitions of the
  listed classes. ProGuard continues processing as usual, only considering the
  first definitions. The warning may be an indication of some problem though,
  so it's advisable to remove the duplicates. A convenient way to do so is by
  specifying filters on the input jars or library jars. You can switch off
  these notes by specifying the [`-dontnote`](usage.md#dontnote) option.

!!! note ""
    ![android](android_small.png){: .icon} The
    standard Android build process automatically specifies the input
    jars for you. There may not be an easy way to filter them to remove
    these notes. You could remove the duplicate classes manually from
    your libraries. You should never explicitly specify the input jars
    yourself (with `-injars` or `-libraryjars`), since you'll then get
    duplicate definitions. You should also not add libraries to your
    application that are already part of the Android run-time (notably
    `org.w3c.dom`, `org.xml.sax`, `org.xmlpull.v1`,
    `org.apache.commons.logging.Log`, `org.apache.http`, and
    `org.json`). They are possibly inconsistent, and the run-time
    libraries would get precedence anyway.

**Warning: can't write resource ... Duplicate zip entry** {: #duplicatezipentry}
: Your input jars contain multiple resource files with the same name.
  ProGuard continues copying the resource files as usual, skipping any files
  with previously used names. Once more, the warning may be an indication of
  some problem though, so it's advisable to remove the duplicates. A
  convenient way to do so is by specifying filters on the input jars. There is
  no option to switch off these warnings.

!!! note ""
    ![android](android_small.png){: .icon} The
    standard Android build process automatically specifies the input
    jars for you. There may not be an easy way to filter them to remove
    these warnings. You could remove the duplicate resource files
    manually from the input and the libraries.

ProGuard may terminate when it encounters parsing errors or I/O errors,
or some more serious warnings:

**Warning: can't find superclass or interface**<br/>**Warning: can't find referenced class** {: #unresolvedclass}
: A class in one of your program jars or library jars is referring to a
  class or interface that is missing from the input. The warning lists both
  the referencing class(es) and the missing referenced class(es). There can be
  a few reasons, with their own solutions:

    1.  If the missing class is referenced from your own code, you may
        have forgotten to specify an essential library. Just like when
        compiling all code from scratch, you must specify all libraries
        that the code is referencing, directly or indirectly. If the
        library should be processed and included in the output, you
        should specify it with [`-injars`](usage.md#injars), otherwise
        you should specify it with [`-libraryjars`](usage.md#libraryjars).
        For example, if ProGuard complains that it can't find a
        `java.lang` class, you have to make sure that you are specifying
        the run-time library of your platform. For JSE, these are
        typically packaged in `lib/rt.jar` (`vm.jar` for IBM's JVM, and
        `classes.jar` in MacOS X) or as of Java 9, `jmods/java.base.jmod`.
        For Android, it is typically packaged in `android.jar`. The
        [examples section](examples.md) provides more details for the
        various platforms. If ProGuard still complains that it can't find a
        `javax.crypto` class, you probably still have to specify `jce.jar`,
        next to the more common `rt.jar`.

    2.  If the missing class is referenced from a pre-compiled
        third-party library, and your original code runs fine without
        it, then the missing dependency doesn't seem to hurt. The
        cleanest solution is to [filter out](usage.md#filters) the
        *referencing* class or classes from the input, with a filter
        like
         "`-injars myapplication.jar(!somepackage/SomeUnusedReferencingClass.class)`".
        DexGuard will then skip this class entirely in the input, and it
        will not bump into the problem of its missing reference.
        However, you may then have to filter out other classes that are
        in turn referencing the removed class. In practice, this works
        best if you can filter out entire unused packages at once, with
        a wildcard filter like
        "`-libraryjars mylibrary.jar(!someunusedpackage/**)`".

    3.  If you don't feel like filtering out the problematic classes,
        you can try your luck with the
        [`-ignorewarnings`](usage.md#ignorewarnings) option, or even
        the [`-dontwarn`](usage.md#dontwarn) option. Only use these
        options if you really know what you're doing though.


!!! note ""
    ![android](android_small.png){: .icon} The
    standard Android build process automatically specifies the input
    jars for you. Unfortunately, many pre-compiled third-party libraries
    refer to other libraries that are not actually used and therefore
    not present. This works fine in debug builds, but in release builds,
    ProGuard expects all libraries, so it can perform a proper
    static analysis. For example, if ProGuard complains that it can't
    find a `java.awt` class, then some library that you are using is
    referring to `java.awt`. This is a bit shady, since Android doesn't
    have this package at all, but if your application works anyway, you
    can let ProGuard accept it with "`-dontwarn java.awt.**`",
    for instance.

    If the missing class is an Android run-time class, you should make
    sure that you are building against an Android run-time that is
    sufficiently recent. You may need to change the build target in your
    `project.properties` file or `build.gradle` file to that
    recent version. You can still specify a different `minSdkVersion`
    and a different `targetSdkVersion` in your
    `AndroidManifest.xml` file.

**Error: Can't find any super classes of ... (not even immediate super class ...)**
**Error: Can't find common super class of ... and ...** {: #superclass}
: It seems like you tried to avoid the warnings from the previous paragraph
  by specifying [`-ignorewarnings`](usage.md#ignorewarnings) or
  [`-dontwarn`](usage.md#dontwarn), but it didn't work out. ProGuard's
  optimization step and preverification step really need the missing classes
  to make sense of the code. Preferably, you would solve the problem by adding
  the missing library, as discussed. If you're sure the class that references
  the missing class isn't used either, you could also try filtering it out
  from the input, by adding a filter to the corresponding
  [`-injars`](usage.md#injars) option: "`-injars
  myapplication.jar(!somepackage/SomeUnusedClass.class)`". As a final
  solution, you could switch off optimization
  ([`-dontoptimize`](usage.md#dontoptimize)) and preverification
  ([`-dontpreverify`](usage.md#dontpreverify)).

**Warning: can't find referenced field/method '...' in program class ...** {: #unresolvedprogramclassmember}
: A program class is referring to a field or a method that is missing from
  another program class. The warning lists both the referencing class and the
  missing referenced class member. Your compiled class files are most likely
  inconsistent. Possibly, some class file didn't get recompiled properly, or
  some class file was left behind after its source file was removed. Try
  removing all compiled class files and rebuilding your project.

**Warning: can't find referenced field/method '...' in library class ...** {: #unresolvedlibraryclassmember}
: A program class is referring to a field or a method that is missing from a
  library class. The warning lists both the referencing class and the missing
  referenced class member. Your compiled class files are inconsistent with the
  libraries. You may need to recompile the class files, or otherwise upgrade
  the libraries to consistent versions.

    1.  If there are unresolved references to class members in *program
        classes*, your compiled class files are most likely
        inconsistent. Possibly, some class file didn't get recompiled
        properly, or some class file was left behind after its source
        file was removed. Try removing all compiled class files and
        rebuilding your project.

    2.  If there are unresolved references to class members in *library
        classes*, your compiled class files are inconsistent with
        the libraries. You may need to recompile the class files, or
        otherwise upgrade the libraries to consistent versions.

    Alternatively, you may get away with ignoring the inconsistency
    with the options [`-ignorewarnings`](usage.md#ignorewarnings)
    or even [`-dontwarn`](usage.md#dontwarn). For instance if the
    code contains a class to optionally support recent versions of
    Android, you can specify "`-dontwarn mypackage.MySupportClass`".

!!! note ""
    ![android](android_small.png){: .icon} If
    If you're developing for Android, and ProGuard complains that it can't
    find a run-time method that is only available in recent versions of
    Android, you should change the target to that recent version in your
    build configuration. You can still specify a different `minSdkVersion`
    and a different `targetSdkVersion` in your `AndroidManifest.xml` file.

**Warning: can't find enclosing class/method** {: #unresolvedenclosingmethod}
: If there are unresolved references to classes that are defined inside
  methods in your input, once more, your compiled class files are most likely
  inconsistent. Possibly, some class file didn't get recompiled properly, or
  some class file was left behind after its source file was removed. Try
  removing all compiled class files and rebuilding your project.

**Warning: library class ... depends on program class ...** {: #dependency}
: If any of your library classes depend on your program classes, by
  extending, implementing or just referencing them, your processed code will
  generally be unusable. Program classes can depend on library classes, but
  not the other way around. Program classes are processed, while library
  classes always remain unchanged. It is therefore impossible to adapt
  references from library classes to program classes, for instance if the
  program classes are renamed. You should define a clean separation between
  program code (specified with [`-injars`](usage.md#injars)) and library code
  (specified with [`-libraryjars`](usage.md#libraryjars)), and try again.

!!! note ""
    ![android](android_small.png){: .icon} In
    Android development, sloppy libraries may contain duplicates of
    classes that are already present in the Android run-time (notably
    `org.w3c.dom`, `org.xml.sax`, `org.xmlpull.v1`,
    `org.apache.commons.logging.Log`, `org.apache.http`, and
    `org.json`). You must remove these classes from your libraries,
    since they are possibly inconsistent, and the run-time libraries
    would get precedence anyway.

**Warning: class file ... unexpectedly contains class ...** {: #unexpectedclass}
: The given class file contains a definition for the given class, but the
  directory name of the file doesn't correspond to the package name of the
  class. ProGuard will accept the class definition, but the current
  implementation will not write out the processed version. Please make sure
  your input classes are packaged correctly. Notably, class files that are in
  the `WEB-INF/classes` directory in a war should be packaged in a jar and put
  in the `WEB-INF/lib` directory. If you don't mind these classes not being
  written to the output, you can specify the
  [`-ignorewarnings`](usage.md#ignorewarnings) option, or even the
  [`-dontwarn`](usage.md#dontwarn) option.

**Warning: ... is not being kept as ..., but remapped to ...** {: #mappingconflict1}
: There is a conflict between a [`-keep`](usage.md#keep) option and the
  mapping file specified with an [`-applymapping`](usage.md#applymapping)
  option, in the obfuscation step. The given class name or class member name
  can't be kept by its original name, as specified in the configuration, but
  it has to be mapped to the other given name, as specified in the mapping
  file. You should adapt your configuration or your mapping file to remove the
  conflict. Alternatively, if you're sure the renaming won't hurt, you can
  specify the [`-ignorewarnings`](usage.md#ignorewarnings) option, or even the
  [`-dontwarn`](usage.md#dontwarn) option.

**Warning: field/method ... can't be mapped to ...** {: #mappingconflict2}
: There is a conflict between some new program code and the mapping file
  specified with an [`-applymapping`](usage.md#applymapping) option, in the
  obfuscation step. The given class member can't be mapped to the given name,
  because it would conflict with another class member that is already being
  mapped to the same name. This can happen if you are performing incremental
  obfuscation, applying an obfuscation mapping file from an initial
  obfuscation step. For instance, some new class may have been added that
  extends two existing classes, introducing a conflict in the name space of
  its class members. If you're sure the class member receiving another name
  than the one specified won't hurt, you can specify the
  [`-ignorewarnings`](usage.md#ignorewarnings) option, or even the
  [`-dontwarn`](usage.md#dontwarn) option. Note that you should always use the
  [`-useuniqueclassmembernames`](usage.md#useuniqueclassmembernames) option in
  the initial obfuscation step, in order to reduce the risk of conflicts.

**Error: Unsupported class version number** {: #unsupportedclassversion}
: You are trying to process class files compiled for a recent version of
  Java that your copy of ProGuard doesn't support yet. You should [check
  on-line](http://proguard.sourceforge.net/downloads.html) if there is a more
  recent release.

**Error: You have to specify [`-keep`](usage.md#keep) options**
: You either forgot to specify [`-keep`](usage.md#keep) options, or you
  mistyped the class names. ProGuard has to know exactly what you want to
  keep: an application, an applet, a servlet, a midlet,..., or any combination
  of these. Without the proper seed specifications, ProGuard would shrink,
  optimize, or obfuscate all class files away.

**Error: Expecting class path separator ';' before 'Files\Java\\**...**'** (in Windows)
: If the path of your run-time jar contains spaces, like in "Program Files",
  you have to enclose it with single or double quotes, as explained in the
  section on [file names](usage.md#filename). This is actually true for all
  file names containing special characters, on all platforms.

**Error: Can't read [**...**/lib/rt.jar\] (No such file or directory)**
: In MacOS X, the run-time classes may be in a different place than on most
  other platforms. You'll then have to adapt your configuration, replacing the
  path `<java.home>/lib/rt.jar` by `<java.home>/../Classes/classes.jar`.

    As of Java 9, the runtime classes are packaged in
    `<java.home>/jmods/java.base.jmod` and other modules next to it.

**Error: Can't read ...** {: #cantread}
: ProGuard can't read the specified file or directory. Double-check that the
  name is correct in your configuration, that the file is readable, and that
  it is not corrupt. An additional message "Unexpected end of ZLIB input
  stream" suggests that the file is truncated. You should then make sure that
  the file is complete on disk when ProGuard starts (asynchronous copying?
  unflushed buffer or cache?), and that it is not somehow overwritten by
  ProGuard's own output.

**Error: Can't write ...** {: #cantwrite}
: ProGuard can't write the specified file or directory. Double-check that
  the name is correct in your configuration and that the file is writable.

**Internal problem starting the ProGuard GUI (Cannot write XdndAware property)** (in Linux)
: In Linux, at least with Java 6, the GUI may not start properly, due to
  [Sun Bug \#7027598](http://bugs.sun.com/view_bug.do?bug_id=7027598). The
  work-around at this time is to specify the JVM option
  `-DsuppressSwingDropSupport=true` when running the GUI.

Should ProGuard crash while processing your application:

**OutOfMemoryError** {: #outofmemoryerror}
: You can try increasing the heap size of the Java virtual machine, with the
  usual `-Xmx` option:

    - In Java, specify the option as an argument to the JVM: `java -Xmx1024m`
    - In Ant, set the environment variable `ANT_OPTS=-Xmx1024m`
    - In Gradle, set the environment variable `GRADLE_OPTS=-Xmx1024m`
    - In Maven, set the environment variable `MAVEN_OPTS=-Xmx1024m`
    - In Eclipse, add the line `-Xmx1024m` to the file `eclipse.ini` inside
      your Eclipse install.

    You can also reduce the amount of memory that ProGuard needs by
    removing unnecessary library jars from your configuration, or by
    filtering out unused library packages and classes.

**StackOverflowError** {: #stackoverflowerror}
: This error might occur when processing a large code base on Windows
  (surprisingly, not so easily on Linux). In theory, increasing the stack size
  of the Java virtual machine (with the usual `-Xss` option) should help too.
  In practice however, the `-Xss` setting doesn't have any effect on the main
  thread, due to [Sun Bug
  \#4362291](http://bugs.sun.com/view_bug.do?bug_id=4362291). As a result,
  this solution will only work when running ProGuard in a different thread,
  e.g. from its GUI.

**Unexpected error** {: #unexpectederror}
: ProGuard has encountered an unexpected condition, typically in the
  optimization step. It may or may not recover. You should be able to avoid it
  using the [`-dontoptimize`](usage.md#dontoptimize) option. In any case,
  please report the problem, preferably with the simplest example that causes
  ProGuard to crash.

**Otherwise...** {: #otherwise}
: Maybe your class files are corrupt. See if recompiling them and trying
  again helps. If not, please report the problem, preferably with the simplest
  example that causes ProGuard to crash.

## Unexpected observations after processing {: #afterprocessing}

If ProGuard seems to run fine, but your processed code doesn't look
right, there might be a couple of reasons:

**Disappearing classes** {: #disappearingclasses}
: If you are working on Windows and it looks like some classes have
  disappeared from your output, you should make sure you're not writing your
  output class files to a directory (or unpacking the output jar). On
  platforms with case-insensitive file systems, such as Windows, unpacking
  tools often let class files with similar lower-case and upper-case names
  overwrite each other. If you really can't switch to a different operating
  system, you could consider using ProGuard's
  [`-dontusemixedcaseclassnames`](usage.md#dontusemixedcaseclassnames) option.
  Also, you should make sure your class files are in directories that
  correspond to their package names. ProGuard will read misplaced class files,
  but it will currently not write their processed versions. Notably, class
  files that are in the `WEB-INF/classes` directory in a war should be
  packaged in a jar and put in the `WEB-INF/lib` directory.

**Classes or class members not being kept** {: #notkept}
: If ProGuard is not keeping the right classes or class members, make sure
  you are using fully qualified class names. If the package name of some class
  is missing, ProGuard won't match the elements that you might be expecting.
  It may help to double-check for typos too. You can use the
  [`-printseeds`](usage.md#printseeds) option to see which elements are being
  kept exactly.

    If you are using marker interfaces to keep other classes, the marker
    interfaces themselves are probably being removed in the shrinking
    step. You should therefore always explicitly keep any marker
    interfaces, with an option like
    "`-keep interface MyMarkerInterface`".

    Similarly, if you are keeping classes based on annotations, you may
    have to avoid that the annotation classes themselves are removed in
    the shrinking step. You should package the annotation classes as a
    library, or explicitly keep them in your program code with an option
    like "`-keep     @interface *`".

**Class names not being obfuscated** {: #classnamesnotobfuscated}
: If the names of some classes in your obfuscated code aren't obfuscated, you
  should first check all your configuration files. Chances are that some
  `-keep` option is preserving the original names. These options may be hiding
  in your own configuration files or in configuration files from libraries.
  For example, some class names mentioned in the Android manifest must always
  be preserved, to avoid compatibility issues when upgrading versions of the
  app. More specifically, the default Android build process automatically keeps
  the names of activities, broadcast receivers and services. You can
  find the underlying reasons in the Google blog ["Things that cannot
  change"](https://android-developers.googleblog.com/2011/06/things-that-cannot-change.html).

**Field names not being obfuscated** {: #fieldnamesnotobfuscated}
: If the names of some fields in your obfuscated code aren't obfuscated, this
  may be due to `-keep` options preserving the original names, for the sake of
  libraries like GSON. Such libraries perform reflection on the fields. If the
  names were obfuscated, the resulting JSON strings would come out obfuscated
  as well, which generally breaks persistence of the data or communication
  with servers.

**Method names not being obfuscated** {: #methodnamesnotobfuscated}
: If the names of some methods in your obfuscated code aren't obfuscated, this
  is most likely because they extend or implement method names in the
  underlying runtime libraries. Since the runtime libraries are not obfuscated,
  any corresponding names in the application code can't be obfuscated either,
  since they must remain consistent.

**Variable names not being obfuscated** {: #variablenamesnotobfuscated}
: If the names of the local variables and parameters in your obfuscated code
  don't look obfuscated, because they suspiciously resemble the names of their
  types, it's probably because the decompiler that you are using is coming up
  with those names. ProGuard's obfuscation step does remove the original names
  entirely, unless you explicitly keep the `LocalVariableTable` or
  `LocalVariableTypeTable` attributes.

## Problems while converting to Android Dalvik bytecode {: #dalvik}

If ProGuard seems to run fine, but the dx tool in the Android SDK
subsequently fails with an error:

**SimException: local variable type mismatch** {: #simexception}
: This error indicates that ProGuard's optimization step has not been able
  to maintain the correct debug information about local variables. This can
  happen if some code is optimized radically. Possible work-arounds: let the
  java compiler not produce debug information (`-g:none`), or let ProGuard's
  obfuscation step remove the debug information again (by *not* keeping the
  attributes `LocalVariableTable` and `LocalVariableTypeTable` with
  [`-keepattributes`](usage.md#keepattributes)), or otherwise just disable
  optimization ([`-dontoptimize`](usage.md#dontoptimize)).

**Conversion to Dalvik format failed with error 1** {: #conversionerror}
: This error may have various causes, but if dx is tripping over some code
  processed by ProGuard, you should make sure that you are using the latest
  version of ProGuard. You can just copy the ProGuard jars to
  `android-sdk/tools/proguard/lib`. If that doesn't help, please report the
  problem, preferably with the simplest example that still brings out the
  error.

## Problems while preverifying for Java Micro Edition

If ProGuard seems to run fine, but the external preverifier subsequently
produces errors, it's usually for a single reason:

**InvalidClassException**, **class loading error**, or **verification error**
: If you get any such message from the preverifier, you are probably working
  on a platform with a case-insensitive file system, such as Windows. The
  `preverify` tool always unpacks the jars, so class files with similar
  lower-case and upper-case names overwrite each other. You can use ProGuard's
  [`-dontusemixedcaseclassnames`](usage.md#dontusemixedcaseclassnames) option
  to work around this problem. If the above doesn't help, there is probably a
  bug in the optimization step of ProGuard. Make sure you are using the latest
  version. You should be able to work around the problem by using the
  [`-dontoptimize`](usage.md#dontoptimize) option. You can check the bug
  database to see if it is a known problem (often with a fix). Otherwise,
  please report it, preferably with the simplest example on which you can find
  ProGuard to fail.

Note that it is no longer necessary to use an external preverifier. With
the [`-microedition`](usage.md#microedition) option, ProGuard will
preverify the class files for Java Micro Edition.

## Problems at run-time {: #runtime}

If ProGuard runs fine, but your processed application doesn't work,
there might be several reasons:

**Stack traces without class names or line numbers** {: #stacktraces}
: If your stack traces don't contain any class names or lines numbers, even
  though you are keeping the proper attributes, make sure this debugging
  information is present in your compiled code to start with. Notably the Ant
  javac task has debugging information switched off by default.

**NoClassDefFoundError** {: #noclassdeffounderror}
: Your class path is probably incorrect. It should at least contain all
  library jars and, of course, your processed program jar.

**ClassNotFoundException** {: #classnotfoundexception}
: Your code is probably calling `Class.forName`, trying to create the
  missing class dynamically. ProGuard can only detect constant name arguments,
  like `Class.forName("com.example.MyClass")`. For variable name arguments
  like `Class.forName(someClass)`, you have to keep all possible classes using
  the appropriate [`-keep`](usage.md#keep) option, e.g. "`-keep class
  com.example.MyClass`" or "`-keep class * implements
  com.example.MyInterface`". While setting up your configuration, you can
  specify the option
  [`-addconfigurationdebugging`](usage.md#addconfigurationdebugging) to help
  track down these cases at run-time and let the instrumented code suggest
  settings for them.

**NoSuchFieldException** {: #nosuchfieldexception}
: Your code is probably calling something like `myClass.getField`, trying to
  find some field dynamically. Since ProGuard can't always detect this
  automatically, you have to keep the missing field using the appropriate
  [`-keep`](usage.md#keep) option, e.g. "`-keepclassmembers class
  com.example.MyClass { int myField; }`". While setting up your configuration,
  you can specify the option
  [`-addconfigurationdebugging`](usage.md#addconfigurationdebugging) to help
  track down these cases at run-time and let the instrumented code suggest
  settings for them.

**NoSuchMethodException** {: #nosuchmethodexception}
: Your code is probably calling something like `myClass.getMethod`, trying to
  find some method dynamically. Since ProGuard can't always detect this
  automatically, you have to keep the missing method using the appropriate
  [`-keep`](usage.md#keep) option, e.g. "`-keepclassmembers class
  com.example.MyClass { void myMethod(); }`". While setting up your
  configuration, you can specify the option
  [`-addconfigurationdebugging`](usage.md#addconfigurationdebugging) to help
  track down these cases at run-time and let the instrumented code suggest
  settings for them. More specifically, if the method reported as missing is
  `values` or `valueOf`, you probably have to keep some methods related to
  [enumerations](examples.md#enumerations).

**MissingResourceException** or **NullPointerException**
: Your processed code may be unable to find some resource files. ProGuard
  simply copies resource files over from the input jars to the output jars.
  Their names and contents remain unchanged, unless you specify the options
  [`-adaptresourcefilenames`](usage.md#adaptresourcefilenames) and/or
  [`-adaptresourcefilecontents`](usage.md#adaptresourcefilecontents).
  Furthermore, directory entries in jar files aren't copied, unless you
  specify the option [`-keepdirectories`](usage.md#keepdirectories). Note that
  Sun advises against calling `Class.getResource()` for directories (Sun Bug
  \#4761949](http://bugs.sun.com/view_bug.do?bug_id=4761949)).

**Disappearing annotations** {: #disappearingannotations}
: By default, the obfuscation step removes all annotations. If your
  application relies on annotations to function properly, you should
  explicitly keep them with `-keepattributes *Annotation*`.

**Invalid or corrupt jarfile** {: #invalidjarfile}
: You are probably starting your application with the java option `-jar`
  instead of the option `-classpath`. The java virtual machine returns with
  this error message if your jar doesn't contain a manifest file
  (`META-INF/MANIFEST.MF`), if the manifest file doesn't specify a main class
  (`Main-Class:` ...), or if the jar doesn't contain this main class. You
  should then make sure that the input jar contains a valid manifest file to
  start with, that this manifest file is the one that is copied (the first
  manifest file that is encountered), and that the main class is kept in your
  configuration,

**InvalidJarIndexException: Invalid index** {: #invalidjarindexexception}
: At least one of your processed jar files contains an index file
  `META-INF/INDEX.LIST`, listing all class files in the jar. ProGuard by
  default copies files like these unchanged. ProGuard may however remove or
  rename classes, thus invalidating the file. You should filter the index file
  out of the input (`-injars in.jar(!META-INF/INDEX.LIST)`) or update the file
  after having applied ProGuard (`jar -i out.jar`).

**InvalidClassException**, **class loading error**, or **verification error** (in Java Micro Edition)
: If you get such an error in Java Micro Edition, you may have forgotten to
  specify the [`-microedition`](usage.md#microedition) option, so the
  processed class files are preverified properly.

**Error: No Such Field or Method**, **Error verifying method** (in a Java Micro Edition emulator)
: If you get such a message in a Motorola or Sony Ericsson phone emulator,
  it's because these emulators don't like packageless classes and/or
  overloaded fields and methods. You can work around it by not using the
  options `-repackageclasses ''` and
  [`-overloadaggressively`](usage.md#overloadaggressively). If you're using
  the JME WTK plugin, you can adapt the configuration
  `proguard/wtk/default.pro` that's inside the `proguard.jar`.

**Failing midlets** (on a Java Micro Edition device)
: If your midlet runs in an emulator and on some devices, but not on some
  other devices, this is probably due to a bug in the latter devices. For some
  older Motorola and Nokia phones, you might try specifying the
  [`-useuniqueclassmembernames`](usage.md#useuniqueclassmembernames) option.
  It avoids overloading class member names, which triggers a bug in their java
  virtual machine. You might also try using the
  [`-dontusemixedcaseclassnames`](usage.md#dontusemixedcaseclassnames) option.
  Even if the midlet has been properly processed and then preverified on a
  case-sensitive file system, the device itself might not like the mixed-case
  class names. Notably, the Nokia N-Gage emulator works fine, but the actual
  device seems to exhibit this problem.

**Disappearing loops** {: #disappearingloops}
: If your code contains empty busy-waiting loops, ProGuard's optimization
  step may remove them. More specifically, this happens if a loop continuously
  checks the value of a non-volatile field that is changed in a different
  thread. The specifications of the Java Virtual Machine require that you
  always mark fields that are accessed across different threads without
  further synchronization as `volatile`. If this is not possible for some
  reason, you'll have to switch off optimization using the
  [`-dontoptimize`](usage.md#dontoptimize) option.

**SecurityException: SHA1 digest error** {: #securityexception}
: You may have forgotten to sign your program jar *after* having processed
  it with ProGuard.

**ClassCastException: class not an enum**<br/>**IllegalArgumentException: class not an enum type** {: #classcastexception}
: You should make sure you're preserving the special methods of enumeration
  types, which the run-time environment calls by introspection. The required
  options are shown in the [examples](examples.md#enumerations).

**ArrayStoreException: sun.reflect.annotation.EnumConstantNotPresentExceptionProxy** {: #arraystoreexception}
: You are probably processing annotations involving enumerations. Again, you
  should make sure you're preserving the special methods of the enumeration
  type, as shown in the examples.

**IllegalArgumentException: methods with same signature but incompatible return types** {: #illegalargumentexception}
: You are probably running some code that has been obfuscated with the
  [`-overloadaggressively`](usage.md#overloadaggressively) option. The class
  `java.lang.reflect.Proxy` can't handle classes that contain methods with the
  same names and signatures, but different return types. Its method
  `newProxyInstance` then throws this exception. You can avoid the problem by
  not using the option.

**CompilerError: duplicate addition** {: #compilererror}
: You are probably compiling or running some code that has been obfuscated
  with the [`-overloadaggressively`](usage.md#overloadaggressively) option.
  This option triggers a bug in `sun.tools.java.MethodSet.add` in Sun's JDK
  1.2.2, which is used for (dynamic) compilation. You should then avoid this
  option.

**ClassFormatError: repetitive field name/signature** {: #classformaterror1}
: You are probably processing some code that has been obfuscated before with
  the [`-overloadaggressively`](usage.md#overloadaggressively) option. You
  should then use the same option again in the second processing round.

**ClassFormatError: Invalid index in LocalVariableTable in class file** {: #classformaterror2}
: If you are keeping the `LocalVariableTable` or `LocalVariableTypeTable`
  attributes, ProGuard's optimizing step is sometimes unable to update them
  consistently. You should then let the obfuscation step remove these
  attributes or disable the optimization step.

**NullPointerException: create returned null** (Dagger)<br/>**IllegalStateException: Module adapter for class ... could not be loaded. Please ensure that code generation was run for this module.**<br/>**IllegalStateException: Could not load class ... needed for binding members/...** {:#dagger}
: Dagger 1 relies on reflection to combine annotated base classes and their
  corresponding generated classes. DexGuard's default configuration already
  preserves the generated classes, but you still preserve the annotated base
  classes in your project-specific configuration. This is explained in some
  more detail in the [Dagger example](examples.md#dagger).

**NoSuchMethodError** or **AbstractMethodError** {: #nosuchmethoderror}
: You should make sure you're not writing your output class files to a
  directory on a platform with a case-insensitive file system, such as
  Windows. Please refer to the section about [disappearing
  classes](#disappearingclasses) for details.

    Furthermore, you should check whether you have specified your
    program jars and library jars properly. Program classes can refer to
    library classes, but not the other way around.

    If all of this seems ok, perhaps there's a bug in ProGuard (gasp!).
    If so, please report it, preferably with the simplest example on
    which you can find ProGuard to fail.

**VerifyError** {: #verifyerror}
: Verification errors when executing a program are almost certainly
  the result of a bug in the optimization step of ProGuard. Make sure
  you are using the latest version. You should be able to work around
  the problem by using the
  [`-dontoptimize`](usage.md#dontoptimize) option. You can check the
  bug database to see if it is a known problem (often with a fix).
  Otherwise, please report it, preferably with the simplest example on
  which ProGuard fails.
