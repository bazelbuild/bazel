## Usage

|            |
|------------|-----------------------------
| Windows:   | `proguard` *options* ...
| Linux/Mac: | `proguard.sh` *options* ...

Typically:

|            |
|------------|-----------------------------
| Windows:   | `proguard @myconfig.pro`
| Linux/Mac: | `proguard.sh @myconfig.pro`

## Options

|                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| [`@`](usage.md#at)[*filename*](usage.md#filename)                                                                                                                        | Short for '`-include` *filename*'.
| [`-include`](usage.md#include) [*filename*](usage.md#filename)                                                                                                           | Read configuration options from the given file.
| [`-basedirectory`](usage.md#basedirectory) [*directoryname*](usage.md#filename)                                                                                          | Specifies the base directory for subsequent relative file names.
| [`-injars`](usage.md#injars) [*class\_path*](usage.md#classpath)                                                                                                         | Specifies the program jars (or apks, aabs, aars, wars, ears, jmods, zips, or directories).
| [`-outjars`](usage.md#outjars) [*class\_path*](usage.md#classpath)                                                                                                       | Specifies the names of the output jars (or apks, aabs, aars, wars, ears, jmods, zips, or directories).
| [`-libraryjars`](usage.md#libraryjars) [*class\_path*](usage.md#classpath)                                                                                               | Specifies the library jars (or apks, aabs, aars, wars, ears, jmods, zips, or directories).
| [`-skipnonpubliclibraryclasses`](usage.md#skipnonpubliclibraryclasses)                                                                                                   | Ignore non-public library classes.
| [`-dontskipnonpubliclibraryclasses`](usage.md#dontskipnonpubliclibraryclasses)                                                                                           | Don't ignore non-public library classes (the default).
| [`-dontskipnonpubliclibraryclassmembers`](usage.md#dontskipnonpubliclibraryclassmembers)                                                                                 | Don't ignore package visible library class members.
| [`-keepdirectories`](usage.md#keepdirectories) \[[*directory\_filter*](usage.md#filters)\]                                                                               | Keep the specified directories in the output jars (or wars, ears, zips, or directories).
| [`-target`](usage.md#target) *version*                                                                                                                                   | Set the given version number in the processed classes.
| [`-forceprocessing`](usage.md#forceprocessing)                                                                                                                           | Process the input, even if the output seems up to date.
| [`-keep`](usage.md#keep) \[[,*modifier*](usage.md#keepoptionmodifiers),...\] [*class\_specification*](usage.md#classspecification)                                       | Preserve the specified classes *and* class members.
| [`-keepclassmembers`](usage.md#keepclassmembers) \[[,*modifier*](usage.md#keepoptionmodifiers),...\] [*class\_specification*](usage.md#classspecification)               | Preserve the specified class members, if their classes are preserved as well.
| [`-keepclasseswithmembers`](usage.md#keepclasseswithmembers) \[[,*modifier*](usage.md#keepoptionmodifiers),...\] [*class\_specification*](usage.md#classspecification)   | Preserve the specified classes *and* class members, if all of the specified class members are present.
| [`-keepnames`](usage.md#keepnames) [*class\_specification*](usage.md#classspecification)                                                                                 | Preserve the names of the specified classes *and* class members (if they aren't removed in the shrinking step).
| [`-keepclassmembernames`](usage.md#keepclassmembernames) [*class\_specification*](usage.md#classspecification)                                                           | Preserve the names of the specified class members (if they aren't removed in the shrinking step).
| [`-keepclasseswithmembernames`](usage.md#keepclasseswithmembernames) [*class\_specification*](usage.md#classspecification)                                               | Preserve the names of the specified classes *and* class members, if all of the specified class members are present (after the shrinking step).
| [`-if`](usage.md#if) [*class\_specification*](usage.md#classspecification)                                                                                               | Specify classes and class members that must be present to activate the subsequent `keep` option.
| [`-printseeds`](usage.md#printseeds) \[[*filename*](usage.md#filename)\]                                                                                                 | List classes and class members matched by the various [`-keep`](usage.md#keep) options, to the standard output or to the given file.
| [`-dontshrink`](usage.md#dontshrink)                                                                                                                                       | Don't shrink the input class files.
| [`-printusage`](usage.md#printusage) \[[*filename*](usage.md#filename)\]                                                                                                 | List dead code of the input class files, to the standard output or to the given file.
| [`-whyareyoukeeping`](usage.md#whyareyoukeeping) [*class\_specification*](usage.md#classspecification)                                                                   | Print details on why the given classes and class members are being kept in the shrinking step.
| [`-dontoptimize`](usage.md#dontoptimize)                                                                                                                                 | Don't optimize the input class files.
| [`-optimizations`](usage.md#optimizations) [*optimization\_filter*](optimizations.md)                                                                                    | The optimizations to be enabled and disabled.
| [`-optimizationpasses`](usage.md#optimizationpasses) *n*                                                                                                                 | The number of optimization passes to be performed.
| [`-assumenosideeffects`](usage.md#assumenosideeffects) [*class\_specification*](usage.md#classspecification)                                                             | Assume that the specified methods don't have any side effects, while optimizing.
| [`-assumenoexternalsideeffects`](usage.md#assumenoexternalsideeffects) [*class\_specification*](usage.md#classspecification)                                             | Assume that the specified methods don't have any external side effects, while optimizing.
| [`-assumenoescapingparameters`](usage.md#assumenoescapingparameters) [*class\_specification*](usage.md#classspecification)                                               | Assume that the specified methods don't let any reference parameters escape to the heap, while optimizing.
| [`-assumenoexternalreturnvalues`](usage.md#assumenoexternalreturnvalues) [*class\_specification*](usage.md#classspecification)                                           | Assume that the specified methods don't return any external reference values, while optimizing.
| [`-assumevalues`](usage.md#assumevalues) [*class\_specification*](usage.md#classspecification)                                                                           | Assume fixed values or ranges of values for primitive fields and methods, while optimizing.
| [`-allowaccessmodification`](usage.md#allowaccessmodification)                                                                                                           | Allow the access modifiers of classes and class members to be modified, while optimizing.
| [`-mergeinterfacesaggressively`](usage.md#mergeinterfacesaggressively)                                                                                                   | Allow any interfaces to be merged, while optimizing.
| [`-dontobfuscate`](usage.md#dontobfuscate)                                                                                                                               | Don't obfuscate the input class files.
| [`-printmapping`](usage.md#printmapping) \[[*filename*](usage.md#filename)\]                                                                                             | Print the mapping from old names to new names for classes and class members that have been renamed, to the standard output or to the given file.
| [`-applymapping`](usage.md#applymapping) [*filename*](usage.md#filename)                                                                                                 | Reuse the given mapping, for incremental obfuscation.
| [`-obfuscationdictionary`](usage.md#obfuscationdictionary) [*filename*](usage.md#filename)                                                                               | Use the words in the given text file as obfuscated field names and method names.
| [`-classobfuscationdictionary`](usage.md#classobfuscationdictionary) [*filename*](usage.md#filename)                                                                     | Use the words in the given text file as obfuscated class names.
| [`-packageobfuscationdictionary`](usage.md#packageobfuscationdictionary) [*filename*](usage.md#filename)                                                                 | Use the words in the given text file as obfuscated package names.
| [`-overloadaggressively`](usage.md#overloadaggressively)                                                                                                                 | Apply aggressive overloading while obfuscating.
| [`-useuniqueclassmembernames`](usage.md#useuniqueclassmembernames)                                                                                                       | Ensure uniform obfuscated class member names for subsequent incremental obfuscation.
| [`-dontusemixedcaseclassnames`](usage.md#dontusemixedcaseclassnames)                                                                                                     | Don't generate mixed-case class names while obfuscating.
| [`-keeppackagenames`](usage.md#keeppackagenames) \[*[package\_filter](usage.md#filters)*\]                                                                               | Keep the specified package names from being obfuscated.
| [`-flattenpackagehierarchy`](usage.md#flattenpackagehierarchy) \[*package\_name*\]                                                                                       | Repackage all packages that are renamed into the single given parent package.
| [`-repackageclasses`](usage.md#repackageclasses) \[*package\_name*\]                                                                                                     | Repackage all class files that are renamed into the single given package.
| [`-keepattributes`](usage.md#keepattributes) \[*[attribute\_filter](usage.md#filters)*\]                                                                                 | Preserve the given optional attributes; typically `Exceptions`, `InnerClasses`, `Signature`, `Deprecated`, `SourceFile`, `SourceDir`, `LineNumberTable`, `LocalVariableTable`, `LocalVariableTypeTable`, `Synthetic`, `EnclosingMethod`, and `*Annotation*`.
| [`-keepparameternames`](usage.md#keepparameternames)                                                                                                                     | Keep the parameter names and types of methods that are kept.
| [`-renamesourcefileattribute`](usage.md#renamesourcefileattribute) \[*string*\]                                                                                          | Put the given constant string in the `SourceFile` attributes.
| [`-adaptclassstrings`](usage.md#adaptclassstrings) \[[*class\_filter*](usage.md#filters)\]                                                                               | Adapt string constants in the specified classes, based on the obfuscated names of any corresponding classes.
| [`-adaptresourcefilenames`](usage.md#adaptresourcefilenames) \[[*file\_filter*](usage.md#filefilters)\]                                                                  | Rename the specified resource files, based on the obfuscated names of the corresponding class files.
| [`-adaptresourcefilecontents`](usage.md#adaptresourcefilecontents) \[[*file\_filter*](usage.md#filefilters)\]                                                            | Update the contents of the specified resource files, based on the obfuscated names of the processed classes.
| [`-dontpreverify`](usage.md#dontpreverify)                                                                                                                               | Don't preverify the processed class files.
| [`-microedition`](usage.md#microedition)                                                                                                                                 | Target the processed class files at Java Micro Edition.
| [`-android`](usage.md#android)                                                                                                                                           | Target the processed class files at Android.
| [`-verbose`](usage.md#verbose)                                                                                                                                           | Write out some more information during processing.
| [`-dontnote`](usage.md#dontnote) \[[*class\_filter*](usage.md#filters)\]                                                                                                 | Don't print notes about potential mistakes or omissions in the configuration.
| [`-dontwarn`](usage.md#dontwarn) \[[*class\_filter*](usage.md#filters)\]                                                                                                 | Don't warn about unresolved references at all.
| [`-ignorewarnings`](usage.md#ignorewarnings)                                                                                                                             | Print warnings about unresolved references, but continue processing anyhow.
| [`-printconfiguration`](usage.md#printconfiguration) \[[*filename*](usage.md#filename)\]                                                                                 | Write out the entire configuration, in traditional ProGuard style, to the standard output or to the given file.
| [`-dump`](usage.md#dump) \[[*filename*](usage.md#filename)\]                                                                                                             | Write out the internal structure of the processed class files, to the standard output or to the given file.
| [`-addconfigurationdebugging`](usage.md#addconfigurationdebugging)                                                                                                       | Instrument the processed code with debugging statements that print out suggestions for missing ProGuard configuration.

Notes:

- *class\_path* is a list of jars, apks, aabs, aars, wars, ears, jmods, zips,
  and directories, with optional filters, separated by path separators.
- *filename* can contain Java system properties delimited by
  '**&lt;**' and '**&gt;**'.
- If *filename* contains special characters, the entire name should be
  quoted with single or double quotes.

## Overview of `Keep` Options {: #keepoverview}

| Keep                                                | From being removed or renamed                                | From being renamed
|-----------------------------------------------------|--------------------------------------------------------------|------------------------------------------------------------------------
| Classes and class members                           | [`-keep`](usage.md#keep)                                     | [`-keepnames`](usage.md#keepnames)
| Class members only                                  | [`-keepclassmembers`](usage.md#keepclassmembers)             | [`-keepclassmembernames`](usage.md#keepclassmembernames)
| Classes and class members, if class members present | [`-keepclasseswithmembers`](usage.md#keepclasseswithmembers) | [`-keepclasseswithmembernames`](usage.md#keepclasseswithmembernames)

## Keep Option Modifiers {: #keepoptionmodifiers}

|                                                                 |
|-----------------------------------------------------------------|---------------------------------------------------------------------------
| [`includedescriptorclasses`](usage.md#includedescriptorclasses) | Also keep any classes in the descriptors of specified fields and methods.
| [`includecode`](usage.md#includecode)                           | Also keep the code of the specified methods unchanged.
| [`allowshrinking`](usage.md#allowshrinking)                     | Allow the specified entry points to be removed in the shrinking step.
| [`allowoptimization`](usage.md#allowoptimization)               | Allow the specified entry points to be modified in the optimization step.
| [`allowobfuscation`](usage.md#allowobfuscation)                 | Allow the specified entry points to be renamed in the obfuscation step.

## Class Specifications {: #classspecification}

    [@annotationtype] [[!]public|final|abstract|@ ...] [!]interface|class|enum classname
        [extends|implements [@annotationtype] classname]
    [{
        [@annotationtype]
        [[!]public|private|protected|static|volatile|transient ...]
        <fields> | (fieldtype fieldname [= values]);

        [@annotationtype]
        [[!]public|private|protected|static|synchronized|native|abstract|strictfp ...]
        <methods> | <init>(argumenttype,...) | classname(argumenttype,...) | (returntype methodname(argumenttype,...));

        [@annotationtype] [[!]public|private|protected|static ... ] *;
        ...
    }]

Notes:

- Class names must always be fully qualified, i.e. including their
  package names.
- Types in *classname*, *annotationtype*, *returntype*, and
  *argumenttype* can contain wildcards: '`?`' for a single character,
  '`*`' for any number of characters (but not the package separator),
  '`**`' for any number of (any) characters, '`%`' for any primitive
  type, '`***`' for any type, '`...`' for any number of arguments, and
  '`<n>`' for the *n*'th matched wildcard in the same option.
- *fieldname* and *methodname* can contain wildcards as well: '`?`'
  for a single character and '`*`' for any number of characters.
