**ProGuard** can be run as a task in the Java-based build tool Gradle
(version 2.1 or higher).

Before you can use the **`proguard`** task, you have to make sure Gradle can
find it in its class path at build time. One way is to add the following
line to your **`build.gradle`** file:

    buildscript {
        repositories {
            jcenter()
        }
        dependencies {
            classpath 'net.sf.proguard:proguard-gradle:6.2.2'
        }
    }

Or if you want to use your local copy of the plugin:

    buildscript {
        repositories {
            flatDir dirs: '/usr/local/java/proguard/lib'
        }
        dependencies {
            classpath ':proguard:'
        }
    }

Please make sure the class path is set correctly for your system.

You can then define a task:

    task myProguardTask(type: proguard.gradle.ProGuardTask) {
        .....
    }

The embedded configuration is much like a standard ProGuard
configuration. Notable similarities and differences:

- Like in ProGuard configurations, we're using all lower-case
  names for the settings.
- The options don't have a dash as prefix.
- Arguments typically have quotes.
- Some settings are specified as named arguments.

You can find some sample build files in the **`examples/gradle`** directory
of the ProGuard distribution.

If you prefer a more verbose configuration derived from the Ant task,
you can import the Ant task as a [Gradle task](#anttask).

## Settings {: #proguard}

The ProGuard task supports the following settings in its closure:

`configuration` [*files*](#file)
: Read and merge options from the given ProGuard configuration files.
  The files are resolved and parsed lazily, during the execution phase.

[**`injars`**](usage.md#injars) [*class\_path*](#classpath)
: Specifies the program jars (or apks, aabs, aars, wars, ears, jmods, zips, or
  directories). The files are resolved and read lazily, during the execution
  phase.

[**`outjars`**](usage.md#outjars) [*class\_path*](#classpath)
: Specifies the names of the output jars (or apks, aabs, aars, wars, ears,
  jmods, zips, or directories). The files are resolved and written lazily,
  during the execution phase.

[**`libraryjars`**](usage.md#libraryjars) [*class\_path*](#classpath)
: Specifies the names of the output jars (or apks, aabs, aars, wars, ears,
  jmods, zips, or directories). The files are resolved and written lazily,
  during the execution phase.

[**`skipnonpubliclibraryclasses`**](usage.md#skipnonpubliclibraryclasses)
: Ignore non-public library classes.

[**`dontskipnonpubliclibraryclassmembers`**](usage.md#dontskipnonpubliclibraryclassmembers)
: Don't ignore package visible library class members.

[**`keepdirectories`**](usage.md#keepdirectories) \['[*directory\_filter*](usage.md#filefilters)'\]
: Keep the specified directories in the output jars (or apks, aabs, aars, wars,
  ears, jmods, zips, or directories).

[**`target`**](usage.md#target) '*version*'
: Set the given version number in the processed classes.

[**`forceprocessing`**](usage.md#forceprocessing)
: Process the input, even if the output seems up to date.

[**`keep`**](usage.md#keep) \[[*modifier*,...](#keepmodifier)\] [*class\_specification*](#classspecification)
: Preserve the specified classes *and* class members.

[**`keepclassmembers`**](usage.md#keepclassmembers) \[[*modifier*,...](#keepmodifier)\] [*class\_specification*](#classspecification)
: Preserve the specified class members, if their classes are preserved as
  well.

[**`keepclasseswithmembers`**](usage.md#keepclasseswithmembers) \[[*modifier*,...](#keepmodifier)\] [*class\_specification*](#classspecification)
: Preserve the specified classes *and* class members, if all of the
  specified class members are present.

[**`keepnames`**](usage.md#keepnames) [*class\_specification*](#classspecification)
: Preserve the names of the specified classes *and* class members (if they
  aren't removed in the shrinking step).

[**`keepclassmembernames`**](usage.md#keepclassmembernames) [*class\_specification*](#classspecification)
: Preserve the names of the specified class members (if they aren't removed
  in the shrinking step).

[**`keepclasseswithmembernames`**](usage.md#keepclasseswithmembernames) [*class\_specification*](#classspecification)
: Preserve the names of the specified classes *and* class members, if all of
  the specified class members are present (after the shrinking step).

[**`printseeds`**](usage.md#printseeds) \[[*file*](#file)\]
: List classes and class members matched by the various **`keep`** commands,
  to the standard output or to the given file.

[**`dontshrink`**](usage.md#dontshrink)
: Don't shrink the input class files.

[**`printusage`**](usage.md#printusage) \[[*file*](#file)\]
: List dead code of the input class files, to the standard output or to the
  given file.

[**`whyareyoukeeping`**](usage.md#whyareyoukeeping) [*class\_specification*](#classspecification)
: Print details on why the given classes and class members are being kept in
  the shrinking step.

[**`dontoptimize`**](usage.md#dontoptimize)
: Don't optimize the input class files.

[**`optimizations`**](usage.md#optimizations) '[*optimization\_filter*](optimizations.md)'
: Perform only the specified optimizations.

[**`optimizationpasses`**](usage.md#optimizationpasses) *n*
: The number of optimization passes to be performed.

[**`assumenosideeffects`**](usage.md#assumenosideeffects) [*class\_specification*](#classspecification)
: Assume that the specified methods don't have any side effects, while
  optimizing. *Only use this option if you know what you're doing!*

[**`assumenoexternalsideeffects`**](usage.md#assumenoexternalsideeffects) [*class\_specification*](#classspecification)
: Assume that the specified methods don't have any external side effects,
  while optimizing. *Only use this option if you know what you're doing!*

[**`assumenoescapingparameters`**](usage.md#assumenoescapingparameters) [*class\_specification*](#classspecification)
: Assume that the specified methods don't let any reference parameters
  escape to the heap, while optimizing. *Only use this option if you know what
  you're doing!*

[**`assumenoexternalreturnvalues`**](usage.md#assumenoexternalreturnvalues) [*class\_specification*](#classspecification)
: Assume that the specified methods don't return any external reference
  values, while optimizing. *Only use this option if you know what you're
  doing!*

[**`assumevalues`**](usage.md#assumevalues) [*class\_specification*](#classspecification)
: Assume fixed values or ranges of values for primitive fields and methods,
  while optimizing. *Only use this option if you know what you're doing!*

[**`allowaccessmodification`**](usage.md#allowaccessmodification)
: Allow the access modifiers of classes and class members to be modified,
  while optimizing.

[**`mergeinterfacesaggressively`**](usage.md#mergeinterfacesaggressively)
: Allow any interfaces to be merged, while optimizing.

[**`dontobfuscate`**](usage.md#dontobfuscate)
: Don't obfuscate the input class files.

[**`printmapping`**](usage.md#printmapping) \[[*file*](#file)\]
: Print the mapping from old names to new names for classes and class
  members that have been renamed, to the standard output or to the given file.

[**`applymapping`**](usage.md#applymapping) [*file*](#file)
: Reuse the given mapping, for incremental obfuscation.

[**`obfuscationdictionary`**](usage.md#obfuscationdictionary) [*file*](#file)
: Use the words in the given text file as obfuscated field names and method
  names.

[**`classobfuscationdictionary`**](usage.md#classobfuscationdictionary) [*file*](#file)
: Use the words in the given text file as obfuscated class names.

[**`packageobfuscationdictionary`**](usage.md#packageobfuscationdictionary) [*file*](#file)
: Use the words in the given text file as obfuscated package names.

[**`overloadaggressively`**](usage.md#overloadaggressively)
: Apply aggressive overloading while obfuscating.

[**`useuniqueclassmembernames`**](usage.md#useuniqueclassmembernames)
: Ensure uniform obfuscated class member names for subsequent incremental
  obfuscation.

[**`dontusemixedcaseclassnames`**](usage.md#dontusemixedcaseclassnames)
: Don't generate mixed-case class names while obfuscating.

[**`keeppackagenames`**](usage.md#keeppackagenames) \['[*package\_filter*](usage.md#filters)'\]
: Keep the specified package names from being obfuscated. If no name is
  given, all package names are preserved.

[**`flattenpackagehierarchy`**](usage.md#flattenpackagehierarchy) '*package\_name*'
: Repackage all packages that are renamed into the single given parent
  package.

[**`repackageclasses`**](usage.md#repackageclasses) \['*package\_name*'\]
: Repackage all class files that are renamed into the single given package.

[**`keepattributes`**](usage.md#keepattributes) \['[*attribute\_filter*](usage.md#filters)'\]
: Preserve the specified optional Java bytecode attributes, with optional
  wildcards. If no name is given, all attributes are preserved.

[**`keepparameternames`**](usage.md#keepparameternames)
: Keep the parameter names and types of methods that are kept.

[**`renamesourcefileattribute`**](usage.md#renamesourcefileattribute) \['*string*'\]
: Put the given constant string in the **`SourceFile`** attributes.

[**`adaptclassstrings`**](usage.md#adaptclassstrings) \['[*class\_filter*](usage.md#filters)'\]
: Adapt string constants in the specified classes, based on the obfuscated
  names of any corresponding classes.

[**`adaptresourcefilenames`**](usage.md#adaptresourcefilenames) \['[*file\_filter*](usage.md#filefilters)'\]
: Rename the specified resource files, based on the obfuscated names of the
  corresponding class files.

[**`adaptresourcefilecontents`**](usage.md#adaptresourcefilecontents) \['[*file\_filter*](usage.md#filefilters)'\]
: Update the contents of the specified resource files, based on the
  obfuscated names of the processed classes.

[**`dontpreverify`**](usage.md#dontpreverify)
: Don't preverify the processed class files if they are targeted at Java
  Micro Edition or at Java 6 or higher.

[**`microedition`**](usage.md#microedition)
: Target the processed class files at Java Micro Edition.

[**`android`**](usage.md#android)
: Target the processed class files at Android.

[**`verbose`**](usage.md#verbose)
: Write out some more information during processing.

[**`dontnote`**](usage.md#dontnote) '[*class\_filter*](usage.md#filters)'
: Don't print notes about classes matching the specified class name filter.

[**`dontwarn`**](usage.md#dontwarn) '[*class\_filter*](usage.md#filters)'
: Don't print warnings about classes matching the specified class name
  filter. *Only use this option if you know what you're doing!*

[**`ignorewarnings`**](usage.md#ignorewarnings)
: Print warnings about unresolved references, but continue processing
  anyhow. *Only use this option if you know what you're doing!*

[**`printconfiguration`**](usage.md#printconfiguration) \[[*file*](#file)\]
: Write out the entire configuration in traditional ProGuard style, to the
  standard output or to the given file. Useful to replace unreadable XML
  configurations.

[**`dump`**](usage.md#dump) \[[*file*](#file)\]
: Write out the internal structure of the processed class files, to the
  standard output or to the given file.

[**`addconfigurationdebugging`**](usage.md#addconfigurationdebugging)
: Adds debugging information to the code, to print out DexGuard
  configuration suggestions at runtime. *Do not use this option in release
  versions.*

## Class Paths {: #classpath}

Class paths are specified as Gradle file collections, which means they
can be specified as simple strings, with **`files(Object)`**, etc.

In addition, they can have ProGuard filters, specified as
comma-separated named arguments after the file:

`filter:` '[*file\_filter*](usage.md#filefilters)'
: An optional filter for all class file names and resource file names that
  are encountered.

`apkfilter:` '[*file\_filter*](usage.md#filefilters)'
: An optional filter for all apk names that are encountered.

`jarfilter:` '[*file\_filter*](usage.md#filefilters)'
: An optional filter for all jar names that are encountered.

`aarfilter:` '[*file\_filter*](usage.md#filefilters)'
: An optional filter for all aar names that are encountered.

`warfilter:` '[*file\_filter*](usage.md#filefilters)'
: An optional filter for all war names that are encountered.

`earfilter:` '[*file\_filter*](usage.md#filefilters)'
: An optional filter for all ear names that are encountered.

`zipfilter:` '[*file\_filter*](usage.md#filefilters)'
: An optional filter for all zip names that are encountered.

## Files {: #file}

Files are specified as Gradle files, which means they can be specified
as simple strings, as File instances, with `file(Object)`, etc.

In Gradle, file names (any strings really) in double quotes can contain
properties or code inside `${...}`. These are automatically expanded.

For example, `"${System.getProperty('java.home')}/lib/rt.jar"` is
expanded to something like `'/usr/local/java/jdk/jre/lib/rt.jar'`.
Similarly, `System.getProperty('user.home')` is expanded to the user's
home directory, and `System.getProperty('user.dir')` is expanded to the
current working directory.

## Keep Modifiers {: #keepmodifier}

The keep settings can have the following named arguments that modify
their behaviors:

[**`if:`**](usage.md#if) [*class\_specification*](#classspecification)
: Specifies classes and class members that must be present to activate the
  keep option.

[**`includedescriptorclasses:`**](usage.md#includedescriptorclasses) *boolean* (default = false)
: Specifies whether the classes of the fields and methods specified in the
  keep tag must be kept as well.

[**`allowshrinking:`**](usage.md#allowshrinking) *boolean* (default = false)
: Specifies whether the entry points specified in the keep tag may be
  shrunk.

[**`allowoptimization:`**](usage.md#allowoptimization) *boolean* (default = false)
: Specifies whether the entry points specified in the keep tag may be
  optimized.

[**`allowobfuscation:`**](usage.md#allowobfuscation) *boolean* (default = false)
: Specifies whether the entry points specified in the keep tag may be
  obfuscated.

Names arguments are comma-separated, as usual.

## Class Specifications {: #classspecification}

A class specification is a template of classes and class members (fields
and methods). There are two alternative ways to specify such a template:

1.  As a string containing a ProGuard class specification. This is
    the most compact and most readable way. The specification looks like
    a Java declaration of a class with fields and methods. For example:

        keep 'public class com.example.MyMainClass {  \
            public static void main(java.lang.String[]);  \
        }'

2.  As a Gradle-style setting: a method call with named arguments and
    a closure. This is more verbose, but it might be useful for
    programmatic specifications. For example:

        keep access: 'public',
             name:   'com.example.MyMainClass', {
                 method access:     'public static',
                        type:       'void',
                        name:       'main',
                        parameters: 'java.lang.String[]'
        }

The [ProGuard class specification](usage.md#classspecification)
is described on the traditional Usage page.

A Gradle-style class specification can have the following named
arguments:

`access:` '*access\_modifiers*'
: The optional access modifiers of the class. Any space-separated list of
  "public", "final", and "abstract", with optional negators "!".

`annotation:` '*annotation\_name*'
: The optional fully qualified name of an annotation of the class, with
  optional wildcards.

`type:` '*type*'
: The optional type of the class: one of "class", "interface", or
  "!interface".

`name:` '*class\_name*'
: The optional fully qualified name of the class, with optional wildcards.

`extendsannotation:` '*annotation\_name*'
: The optional fully qualified name of an annotation of the the class that
  the specified classes must extend, with optional wildcards.

`'extends':` '*class\_name*'
: The optional fully qualified name of the class the specified classes must
  extend, with optional wildcards.

`'implements':` '*class\_name*'
: The optional fully qualified name of the class the specified classes must
  implement, with optional wildcards.

The named arguments are optional. Without any arguments, there are no
constraints, so the settings match all classes.

## Gradle-style Class Member Specifications {: #classmemberspecification}

The closure of a Gradle-style class specification can specify class
members with these settings:

`field` *field\_constraints*
: Specifies a field.

`method` *method\_constraints*
: Specifies a method.

`constructor` *constructor\_constraints*
: Specifies a constructor.

A class member setting can have the following named arguments to express
constraints:

`access:` '*access\_modifiers*'
: The optional access modifiers of the class. Any space-separated list of
  "public", "protected", "private", "static", etc., with optional negators
  "!".

`'annotation':` '*annotation\_name*'
: The optional fully qualified name of an annotation of the class member,
  with optional wildcards.

`type:` '*type*'
: The optional fully qualified type of the class member, with optional
  wildcards. Not applicable for constructors, but required for methods for
  which the **`parameters`** argument is specified.

`name:` '*name*'
: The optional name of the class member, with optional wildcards. Not
  applicable for constructors.

`parameters:` '*parameters*'
: The optional comma-separated list of fully qualified method parameters,
  with optional wildcards. Not applicable for fields, but required for
  constructors, and for methods for which the **`type`** argument is
  specified.

The named arguments are optional. Without any arguments, there are no
constraints, so the settings match all constructors, fields, or methods.

A class member setting doesn't have a closure.

## Alternative: imported Ant task {: #anttask}

Instead of using the Gradle task, you could also integrate the Ant task
in your Gradle build file:

    ant.project.basedir = '../..'

    ant.taskdef(resource:  'proguard/ant/task.properties',
                classpath: '/usr/local/java/proguard/lib/proguard.jar')

Gradle automatically converts the elements and attributes to Groovy
methods, so converting the configuration is essentially mechanical. The
one-on-one mapping can be useful, but the resulting configuration is
more verbose. For instance:

    task proguard << {
      ant.proguard(printmapping: 'proguard.map',
                   overloadaggressively: 'on',
                   repackageclasses: '',
                   renamesourcefileattribute: 'SourceFile') {

        injar(file: 'application.jar')
        injar(file: 'gui.jar', filter: '!META-INF/**')

        .....
      }
    }
