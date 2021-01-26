You can find the ProGuard GUI jar in the `lib` directory of the ProGuard
distribution. To run the ProGuard graphical user interface, just type:

`proguardgui [-nosplash]` \[*configuration\_file*\]

Alternatively, the `bin` directory contains some short Linux and Windows
scripts containing this command. The GUI will pop up in a window. With the
`-nosplash` option, you can switch off the short opening animation. If you
have specified a ProGuard configuration file, it will be loaded. The GUI works
like a wizard. You can edit the configuration and execute ProGuard through a
few tabs:

|                               |
|-------------------------------|--------------------------------------------------------------
| [ProGuard](#proguard)         | Optionally load an existing configuration file.
| [Input/Output](#inputoutput)  | Specify the program jars and library jars.
| [Shrinking](#shrinking)       | Specify the shrinking options.
| [Obfuscation](#obfuscation)   | Specify the obfuscation options.
| [Optimization](#optimization) | Specify the optimization options.
| [Information](#information)   | Specify some options to get information.
| [Process](#process)           | View and save the resulting configuration, and run ProGuard.

In addition, there is a tab to execute ReTrace interactively:

|                     |
|---------------------|-------------------------------------------------------
| [ReTrace](#retrace) | Set up and run ReTrace, to de-obfuscate stack traces.

You can freely toggle between the tabs by means of the buttons on the
left-hand side of the window, or by means of the **Previous** and **Next**
buttons at the bottom of the tabs. Tool tips briefly explain the purpose of
the numerous options and text fields, although a basic understanding of the
shrinking/optimization/obfuscation/preverification process is assumed. Please
refer to the [Introduction](index.md) of this manual.

## The ProGuard tab {: #proguard}

The *ProGuard* tab presents a welcome message and one important button
at the bottom:

|                       |
|-----------------------|-----------------------------------------------------------------------
| Load configuration... | opens a file chooser to load an existing ProGuard configuration file.

If you don't want to load an existing configuration, you can just continue
creating a new configuration from scratch.

## The Input/Output tab {: #inputoutput}

The *Input/Output* tab contains two lists, respectively to specify the program
jars (or aars, wars, ears, zips, apks, or directories), and the library jars
(or aars, wars, ears, zips, apks, or directories).

- The list of program jars contains input entries and output entries. Input
  entries contain the class files and resource files to be processed. Output
  entries specify the destinations to which the processed results will be
  written. They are preceded by arrows, to distinguish them from input
  entries. The results of each consecutive list of input entries will be
  written to the subsequent consecutive list of output entries.
- The library jars are not copied to the output jars; they contain class files
  that are used by class files in the program jars and that are necessary for
  correct processing. This list typically at least contains the targeted Java
  runtime jar.

Each of these lists can be edited by means of a couple of buttons on the
right-hand side:

|                   |
|-------------------|------------------------------------------------------------------------------------------
| Add input...      | opens a file chooser to add an input entry to the list of program jars.
| Add output...     | opens a file chooser to add an output entry to the list of program jars.
| Add...            | opens a file chooser to add an entry to the list of library jars.
| Edit...           | opens a file chooser to edit the selected entry in the list.
| Filter...         | opens a text entry field to add or edit the filters of the selected entries in the list.
| Remove            | removes the selected entries from the list.
| Move up           | moves the selected entries one position up the list.
| Move down         | moves the selected entries one position down the list.
| Move to libraries | moves the selected entries in the list of program jars to the list of library jars.
| Move to program   | moves the selected entries in the list of library jars to the list of program jars.

Filters allow to filter files based on their names. You can specify filters
for class file names and resource file names, for jar file names, for aar file
names, for war file names, for ear file names, for zip file names, and for apk
file names. Multiple entries in the program list only make sense when combined
with filters; each output file is written to the first entry with a matching
filter.

Input entries that are currently not readable are colored red.

The order of the entries in each list may matter, as the first occurrence of
any duplicate entries gets precedence, just as in conventional class paths.

Corresponding configuration options:

- [-injars](usage.md#injars)
- [-outjars](usage.md#outjars)
- [-libraryjars](usage.md#libraryjars)
- [*class\_path*](usage.md#classpath)
- [*filters*](usage.md#filters)

## The Shrinking tab {: #shrinking}

The *Shrinking* tab presents a number of options that affect the shrinking
step. The basic options are followed by a few lists of classes and class
members (fields and methods) that must be protected from shrinking (and
implicitly from obfuscation as well).

The fixed lists contain predefined entries that are typically useful for many
applications. Each of these entries can be toggled by means of a check box.
The text field following each entry allows to constrain the applicable classes
by means of a comma-separated list of wildcarded, fully-qualified class names.
The default is "\*", which means that all input classes of the corresponding
type are considered.

For example, checking the **Applications** entry and filling in
"myapplications.\*\*" after it would mean: keep all classes that have main
methods in the "myapplications" package and all of its subpackages.

The variable list at the bottom allows to define additional entries yourself.
The list can be edited by means of a couple of buttons on the right-hand side:

|           |
|-----------|--------------------------------------------------------
| Add...    | opens a window to add a new entry to the list.
| Edit...   | opens a window to edit the selected entry in the list.
| Remove    | removes the selected entries from the list.
| Move up   | moves the selected entries one position up the list.
| Move down | moves the selected entries one position down the list.

The interface windows allow to specify classes, fields, and methods.
They contain text fields and check boxes to constrain these items. They
have **Ok** and **Cancel** buttons to apply or to cancel the operation.

For example, your application may be creating some classes dynamically using
`Class.forName`. You should then specify them here, so they are kept by their
original names. Press the **Add...** button to open the class window. Fill out
the fully-qualified class name in the **Code** text field, and press the
**Ok** button. Repeat this for all required classes. Wildcards can be helpful
to specify a large number of related classes in one go. If you want to specify
all implementations of a certain interface, fill out the fully qualified
interface name in the **Extends/implements class** instead.

For more advanced settings, it is advisable to become familiar with ProGuard's
configuration options through the [Usage section](usage.md) and the [Examples
section](examples.md). We'll suffice with a brief overview of the three
dialogs provided by the GUI.

The *keep class* dialog appears when adding or editing new special keep
entries. It has text fields and selections for specifying and constraining
classes and class members to keep. The **Advanced options** / **Basic
options** button at the bottom of the dialog allows to toggle showing the
advanced options.

- The **Comments** text field allows to add optional comments to this entry.
  The comments will identify the entry in the list and they will appear as
  comments in the configuration file.
- The **Keep** selection allows to specify whether you want to protect the
  specified classes and their specified class members, or just the specified
  class members from the specified classes, or the specified classes and the
  specified class members, if the class members are present. Note that class
  members will only be protected if they are explicitly specified, even if
  only by means of a wildcard.
- The **Allow** selection allows to specify whether you want to allow the the
  specified classes and their specified class members to be shrunk, optimized
  and/or obfuscated.
- The **Access** selections allows to specify constraints on the class or
  classes, based on their access modifiers.
- The **Annotation** text field takes the fully-qualified name of an
  annotation that is required for matching classes. The annotation name can
  contain wildcards. This is an advanced option for defining *keep*
  annotations.
- The **Class** text field takes the fully-qualified name of the class or
  classes. The class name can contain wildcards.
- The **Annotation** text field takes the fully-qualified name of an
  annotation that is required for the class or interface that the above class
  must extend. The annotation name can contain wildcards. This is an advanced
  option for defining *keep* annotations.
- The **Extends/implements class** text field takes the fully-qualified name
  of the class or interface that the above classes must extend.
- The **Class members** list allows to specify a list of fields and methods to
  keep. It can be edited by means of a list of buttons on the right-hand side.

The *keep field* dialog appears when adding or editing fields within the above
dialog. It has text fields and selections for specifying and constraining
fields to keep. Again, the **Advanced options** / **Basic options** button at
the bottom of the dialog allows to toggle showing the advanced options.

- The **Access** selections allows to specify constraints on the field or
  fields, based on their access modifiers.
- The **Annotation** text field takes the fully-qualified name of an
  annotation that is required for matching fields. The annotation name can
  contain wildcards. This is an advanced option for defining *keep*
  annotations.
- The **Return type** text field takes the fully-qualified type of the field
  or fields. The type can contain wildcards.
- The **Name** text field takes the name of the field or fields. The field
  name can contain wildcards.

Similarly, the *keep method* dialog appears when adding or editing methods
within the keep class dialog. It has text fields and selections for specifying
and constraining methods to keep. Again, the **Advanced options** / **Basic
options** button at the bottom of the dialog allows to toggle showing the
advanced options.

- The **Access** selections allows to specify constraints on the method or
  methods, based on their access modifiers.
- The **Annotation** text field takes the fully-qualified name of an
  annotation that is required for matching methods. The annotation name can
  contain wildcards. This is an advanced option for defining *keep*
  annotations.
- The **Return type** text field takes the fully-qualified type of the method
  or methods. The type can contain wildcards.
- The **Name** text field takes the name of the method or methods. The method
  name can contain wildcards.
- The **Arguments** text field takes the comma-separated list of
  fully-qualified method arguments. Each of these arguments can contain
  wildcards.

Corresponding configuration options:

- [-dontshrink](usage.md#dontshrink)
- [-printusage](usage.md#printusage)
- [-keep](usage.md#keep)
- [-keepclassmembers](usage.md#keepclassmembers)
- [-keepclasseswithmembers](usage.md#keepclasseswithmembers)

## The Obfuscation tab {: #obfuscation}

The *Obfuscation* tab presents a number of options that affect the obfuscation
step. The basic options are followed by a few lists of classes and class
members (fields and methods) that must be protected from obfuscation (but not
necessarily from shrinking).

The lists are manipulated in the same way as in the [Shrinking
tab](#shrinking).

Corresponding configuration options:

- [-dontobfuscate](usage.md#dontobfuscate)
- [-printmapping](usage.md#printmapping)
- [-applymapping](usage.md#applymapping)
- [-obfuscationdictionary](usage.md#obfuscationdictionary)
- [-classobfuscationdictionary](usage.md#classobfuscationdictionary)
- [-packageobfuscationdictionary](usage.md#packageobfuscationdictionary)
- [-overloadaggressively](usage.md#overloadaggressively)
- [-useuniqueclassmembernames](usage.md#useuniqueclassmembernames)
- [-dontusemixedcaseclassnames](usage.md#dontusemixedcaseclassnames)
- [-keeppackagenames](usage.md#keeppackagenames)
- [-flattenpackagehierarchy](usage.md#flattenpackagehierarchy)
- [-repackageclasses](usage.md#repackageclasses)
- [-keepattributes](usage.md#keepattributes)
- [-keepparameternames](usage.md#keepparameternames)
- [-renamesourcefileattribute](usage.md#renamesourcefileattribute)
- [-adaptclassstrings](usage.md#adaptclassstrings)
- [-adaptresourcefilenames](usage.md#adaptresourcefilenames)
- [-adaptresourcefilecontents](usage.md#adaptresourcefilecontents)
- [-keepnames](usage.md#keepnames)
- [-keepclassmembernames](usage.md#keepclassmembernames)
- [-keepclasseswithmembernames](usage.md#keepclasseswithmembernames)
- [*class\_specification*](usage.md#classspecification)

## The Optimization tab {: #optimization}

The *Optimization* tab presents a number of options that affect the
optimization step. The basic options are followed by a few lists of class
method calls that can be removed if ProGuard can determine that their results
are not being used.

The lists are manipulated in much the same way as in the [Shrinking
tab](#shrinking).

Corresponding configuration options:

- [-dontoptimize](usage.md#dontoptimize)
- [-optimizations](usage.md#optimizations)
- [-optimizationpasses](usage.md#optimizationpasses)
- [-allowaccessmodification](usage.md#allowaccessmodification)
- [-mergeinterfacesaggressively](usage.md#mergeinterfacesaggressively)
- [-assumenosideeffects](usage.md#assumenosideeffects)
- [*class\_specification*](usage.md#classspecification)

## The Information tab {: #information}

The *Information* tab presents a number of options for preverification and
targeting, and for the information that ProGuard returns when processing your
code. The bottom list allows you to query ProGuard about why given classes and
class members are being kept in the shrinking step.

Corresponding configuration options:

- [-dontpreverify](usage.md#dontpreverify)
- [-microedition](usage.md#microedition)
- [-target](usage.md#target)
- [-verbose](usage.md#verbose)
- [-dontnote](usage.md#dontnote)
- [-dontwarn](usage.md#dontwarn)
- [-ignorewarnings](usage.md#ignorewarnings)
- [-skipnonpubliclibraryclasses](usage.md#skipnonpubliclibraryclasses)
- [-dontskipnonpubliclibraryclasses](usage.md#dontskipnonpubliclibraryclasses)
- [-dontskipnonpubliclibraryclassmembers](usage.md#dontskipnonpubliclibraryclassmembers)
- [-keepdirectories](usage.md#keepdirectories)
- [-forceprocessing](usage.md#forceprocessing)
- [-printseeds](usage.md#printseeds)
- [-printconfiguration](usage.md#printconfiguration)
- [-dump](usage.md#dump)
- [-whyareyoukeeping](usage.md#whyareyoukeeping)

## The Process tab {: #process}

The *Process* tab has an output console for displaying the configuration and
the messages while processing. There are three important buttons at the
bottom:

|                       |
|-----------------------|------------------------------------------------------------------
| View configuration    | displays the current ProGuard configuration in the console.
| Save configuration... | opens a file chooser to save the current ProGuard configuration.
| Process!              | executes ProGuard with the current configuration.

## The ReTrace tab {: #retrace}

The *ReTrace* tab has a panel with a few settings, an input text area for the
obfuscated stack trace, and an output console to view the de-obfuscated stack
trace:

- The **Verbose** check box in the settings panel allows to toggle between
  normal mode and verbose mode.
- The **Mapping file** text field takes the name of the required mapping file
  that ProGuard wrote while processing the original code. The file name can be
  entered manually or by means of the **Browse...** button that opens a file
  chooser.
- The **Obfuscated stack trace** text area allows to enter the stack trace,
  typically by copying and pasting it from elsewhere. Alternatively, it can be
  loaded from a file by means of the load button below.

There are two buttons at the bottom:

|                     |
|---------------------|---------------------------------------------------------
| Load stack trace... | opens a file chooser to load an obfuscated stack trace.
| ReTrace!            | executes ReTrace with the current settings.
