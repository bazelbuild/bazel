You can find some sample configuration files in the `examples` directory
of the ProGuard distribution.

## Processing different types of applications {: #applicationtypes}

### A typical application {: #application}

To shrink, optimize, and obfuscate a simple Java application, you
typically create a configuration file like `myconfig.pro`, which you can
then use with

    bin/proguard @myconfig.pro

The configuration file specifies the input, the output, and the entry
points of the application:

    -injars       myapplication.jar
    -outjars      myapplication_out.jar
    -libraryjars  <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)
    -printmapping myapplication.map

    -keep public class com.example.MyMain {
        public static void main(java.lang.String[]);
    }

Note the use of the `<java.home>` system property. ProGuard automatically
replaces it when parsing the file. In this example, the library jar is the
base Java runtime module, minus some unwanted files. For Java 8 or older, the
Java runtime jar would be `<java.home>/lib/rt.jar` instead. You may need
additional modules or jars if your application depends on them.

The [`-keep`](usage.md#keep) option specifies the entry point of the
application that has to be preserved. The access modifiers `public` and
`static` are not really required in this case, since we know a priori that the
specified class and method have the proper access flags. It just looks more
familiar this way.

Note that all type names are fully specified: `com.example.MyMain` and
`java.lang.String[]`.

We're writing out an obfuscation mapping file with
[`-printmapping`](usage.md#printmapping), for de-obfuscating any stack traces
later on, or for incremental obfuscation of extensions.

We can further improve the results with a few additional options:

    -optimizationpasses 3
    -overloadaggressively
    -repackageclasses ''
    -allowaccessmodification

These options are not required; they just shave off some extra bytes from the
output jar, by performing up to 3 optimization passes, and by aggressively
obfuscating class members and [package names](#repackaging).

In general, you might need a few additional options for processing [native
methods](#native), [callback methods](#callback),
[enumerations](#enumerations), [serializable classes](#serializable), [bean
classes](#beans), [annotations](#annotations), and [resource
files](#resourcefiles).

### A typical applet {: #applet}

These options shrink, optimize, and obfuscate the applet
`com.example.MyApplet`:

    -injars      in.jar
    -outjars     out.jar
    -libraryjars <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)

    -keep public class com.example.MyApplet

The typical applet methods will be preserved automatically, since
`com.example.MyApplet` is an extension of the `Applet` class in the library
`rt.jar`.

If applicable, you should add options for processing [native
methods](#native), [callback methods](#callback),
[enumerations](#enumerations), [serializable classes](#serializable), [bean
classes](#beans), [annotations](#annotations), and [resource
files](#resourcefiles).

### A typical midlet {: #midlet}

These options shrink, optimize, obfuscate, and preverify the midlet
`com.example.MyMIDlet`:

    -injars      in.jar
    -outjars     out.jar
    -libraryjars /usr/local/java/wtk2.5.2/lib/midpapi20.jar
    -libraryjars /usr/local/java/wtk2.5.2/lib/cldcapi11.jar
    -overloadaggressively
    -repackageclasses ''
    -allowaccessmodification
    -microedition

    -keep public class com.example.MyMIDlet

Note how we're now targeting the Java Micro Edition run-time environment of
`midpapi20.jar` and `cldcapi11.jar`, instead of the Java Standard Edition
run-time environment `rt.jar`. You can target other JME environments by
picking the appropriate jars.

The typical midlet methods will be preserved automatically, since
`com.example.MyMIDlet` is an extension of the `MIDlet` class in the library
`midpapi20.jar`.

The [`-microedition`](usage.md#microedition) option makes sure the class files
are preverified for Java Micro Edition, producing compact `StackMap`
attributes. It is no longer necessary to run an external preverifier.

Be careful if you do use the external `preverify` tool on a platform with a
case-insensitive filing system, such as Windows. Because this tool unpacks
your processed jars, you should then use ProGuard's
[`-dontusemixedcaseclassnames`](usage.md#dontusemixedcaseclassnames) option.

If applicable, you should add options for processing [native methods](#native)
and [resource files](#resourcefiles).

Note that you will still have to adapt the midlet jar size in the
corresponding jad file; ProGuard doesn't do that for you.

### A typical Java Card applet {: #jcapplet}

These options shrink, optimize, and obfuscate the Java Card applet
`com.example.MyApplet`:

    -injars      in.jar
    -outjars     out.jar
    -libraryjars /usr/local/java/javacard2.2.2/lib/api.jar
    -dontwarn    java.lang.Class
    -overloadaggressively
    -repackageclasses ''
    -allowaccessmodification

    -keep public class com.example.MyApplet

The configuration is very similar to the configuration for midlets, except
that it now targets the Java Card run-time environment. This environment
doesn't have java.lang.Class, so we're telling ProGuard not to worry about it.

### A typical xlet {: #xlet}

These options shrink, optimize, and obfuscate the xlet `com.example.MyXlet`:

    -injars      in.jar
    -outjars     out.jar
    -libraryjars /usr/local/java/jtv1.1/javatv.jar
    -libraryjars /usr/local/java/cdc1.1/lib/cdc.jar
    -libraryjars /usr/local/java/cdc1.1/lib/btclasses.zip
    -overloadaggressively
    -repackageclasses ''
    -allowaccessmodification

    -keep public class com.example.MyXlet

The configuration is very similar to the configuration for midlets, except
that it now targets the CDC run-time environment with the Java TV API.

### A simple Android activity {: #simpleandroid}

These options shrink, optimize, and obfuscate the single Android activity
`com.example.MyActivity`:

    -injars      bin/classes
    -outjars     bin/classes-processed.jar
    -libraryjars /usr/local/java/android-sdk/platforms/android-9/android.jar

    -android
    -dontpreverify
    -repackageclasses ''
    -allowaccessmodification
    -optimizations !code/simplification/arithmetic

    -keep public class com.example.MyActivity

We're targeting the Android run-time and keeping the activity as an entry
point.

Preverification is irrelevant for the dex compiler and the Dalvik VM, so we
can switch it off with the [`-dontpreverify`](usage.md#dontpreverify) option.

The [`-optimizations`](usage.md#optimizations) option disables some arithmetic
simplifications that Dalvik 1.0 and 1.5 can't handle. Note that the Dalvik VM
also can't handle [aggressive overloading](usage.md#overloadaggressively) (of
static fields).

If applicable, you should add options for processing [native
methods](#native), [callback methods](#callback),
[enumerations](#enumerations), [annotations](#annotations), and [resource
files](#resourcefiles).

### A complete Android application {: #android}

!!! note ""
    ![android](android_small.png){: .icon} The standard build processes of the
    Android SDK (with Ant, Gradle, Android Studio, and Eclipse) already
    integrate ProGuard with all the proper settings. You only need to enable
    ProGuard by uncommenting the line "`proguard.config=.....`" in the file
    `project.properties` of your Ant project, or by adapting the
    `build.gradle` file of your Gradle project. You then *don't* need any of
    the configuration below.

Notes:

- In case of problems, you may want to check if the configuration files that
  are listed on this line (`proguard-project.txt`,...) contain the necessary
  settings for your application.
- Android SDK revision 20 and higher have a different configuration file for
  enabling optimization:
  `${sdk.dir}/tools/proguard/proguard-android-optimize.txt` instead of the
  default `${sdk.dir}/tools/proguard/proguard-android.txt`.
- The build processes are already setting the necessary program jars, library
  jars, and output jars for you — don't specify them again.
- If you get warnings about missing referenced classes: it's all too common
  that libraries refer to missing classes. See ["Warning: can't find
  referenced class"](troubleshooting.md#unresolvedclass) in the
  Troubleshooting section.

For more information, you can consult the official [Developer
Guide](http://developer.android.com/guide/developing/tools/proguard.html) in
the Android SDK.

If you're constructing a build process *from scratch*: these options
shrink, optimize, and obfuscate all public activities, services,
broadcast receivers, and content providers from the compiled classes and
external libraries:

    -injars      bin/classes
    -injars      bin/resources.ap_
    -injars      libs
    -outjars     bin/application.apk
    -libraryjars /usr/local/android-sdk/platforms/android-28/android.jar

    -android
    -dontpreverify
    -repackageclasses ''
    -allowaccessmodification
    -optimizations !code/simplification/arithmetic
    -keepattributes *Annotation*

    -keep public class * extends android.app.Activity
    -keep public class * extends android.app.Application
    -keep public class * extends android.app.Service
    -keep public class * extends android.content.BroadcastReceiver
    -keep public class * extends android.content.ContentProvider

    -keep public class * extends android.view.View {
        public <init>(android.content.Context);
        public <init>(android.content.Context, android.util.AttributeSet);
        public <init>(android.content.Context, android.util.AttributeSet, int);
        public void set*(...);
    }

    -keepclasseswithmembers class * {
        public <init>(android.content.Context, android.util.AttributeSet);
    }

    -keepclasseswithmembers class * {
        public <init>(android.content.Context, android.util.AttributeSet, int);
    }

    -keepclassmembers class * extends android.content.Context {
       public void *(android.view.View);
       public void *(android.view.MenuItem);
    }

    -keepclassmembers class * implements android.os.Parcelable {
        static ** CREATOR;
    }

    -keepclassmembers class **.R$* {
        public static <fields>;
    }

    -keepclassmembers class * {
        @android.webkit.JavascriptInterface <methods>;
    }

Most importantly, we're keeping all fundamental classes that may be referenced
by the `AndroidManifest.xml` file of the application. If your manifest file
contains other classes and methods, you may have to specify those as well.

We're keeping annotations, since they might be used by custom `RemoteViews`
and by various frameworks.

We're keeping any custom `View` extensions and other classes with typical
constructors, since they might be referenced from XML layout files.

We're also keeping possible `onClick` handlers in custom `Context` extensions,
since they might be referenced from XML layout files.

We're also keeping the required static fields in `Parcelable` implementations,
since they are accessed by introspection.

We're keeping the static fields of referenced inner classes of auto-generated
`R` classes, just in case your code is accessing those fields by
introspection. Note that the compiler already inlines primitive fields, so
ProGuard can generally remove all these classes entirely anyway (because the
classes are not referenced and therefore not required).

Finally, we're keeping annotated Javascript interface methods, so they can be
exported and accessed by their original names. Javascript interface methods
that are not annotated (in code targeted at Android versions older than 4.2)
still need to be preserved manually.

If you're using additional Google APIs, you'll have to specify those as
well, for instance:

    -libraryjars /usr/local/java/android-sdk/extras/android/support/v4/android-support-v4.jar
    -libraryjars /usr/local/java/android-sdk/add-ons/addon-google_apis-google-21/libs/maps.jar

If you're using Google's optional License Verification Library, you can
obfuscate its code along with your own code. You do have to preserve its
`ILicensingService` interface for the library to work:

    -keep public interface com.android.vending.licensing.ILicensingService

If you're using the Android Compatibility library, you should add the
following line, to let ProGuard know it's ok that the library references some
classes that are not available in all versions of the API:

    -dontwarn android.support.**

If applicable, you should add options for processing [native
methods](#native), [callback methods](#callback),
[enumerations](#enumerations), and [resource files](#resourcefiles). You may
also want to add options for producing [useful stack traces](#stacktrace) and
to [remove logging](#logging). You can find a complete sample configuration in
`examples/standalone/android.pro` in the ProGuard distribution.

### A typical library {: #library}

These options shrink, optimize, and obfuscate an entire library, keeping
all public and protected classes and class members, native method names,
and serialization code. The processed version of the library can then
still be used as such, for developing code based on its public API.

    -injars       in.jar
    -outjars      out.jar
    -libraryjars  <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)
    -printmapping out.map

    -keep public class * {
        public protected *;
    }

    -keepparameternames
    -renamesourcefileattribute SourceFile
    -keepattributes Exceptions,InnerClasses,Signature,Deprecated,
                    SourceFile,LineNumberTable,*Annotation*,EnclosingMethod

    -keepclasseswithmembernames,includedescriptorclasses class * {
        native <methods>;
    }

    -keepclassmembers,allowoptimization enum * {
        public static **[] values();
        public static ** valueOf(java.lang.String);
    }

    -keepclassmembers class * implements java.io.Serializable {
        static final long serialVersionUID;
        private static final java.io.ObjectStreamField[] serialPersistentFields;
        private void writeObject(java.io.ObjectOutputStream);
        private void readObject(java.io.ObjectInputStream);
        java.lang.Object writeReplace();
        java.lang.Object readResolve();
    }

This configuration should preserve everything a developers ever wants to
access in the library. Only if there are any other non-public classes or
methods that are invoked dynamically, they should be specified using
additional [`-keep`](usage.md#keep) options.

The "Exceptions" attribute has to be preserved, so the compiler knows
which exceptions methods may throw.

The "InnerClasses" attribute (or more precisely, its source name part) has to
be preserved too, for any inner classes that can be referenced from outside
the library. The `javac` compiler would be unable to find the inner classes
otherwise.

The "Signature" attribute is required to be able to access generic types when
compiling in JDK 5.0 and higher.

The [`-keepparameternames`](usage.md#keepparameternames) option keeps the
parameter names in the "LocalVariableTable" and "LocalVariableTypeTable"
attributes of public library methods. Some IDEs can present these names to the
developers who use the library.

Finally, we're keeping the "Deprecated" attribute and the attributes for
producing [useful stack traces](#stacktrace).

We've also added some options for for processing [native methods](#native),
[enumerations](#enumerations), [serializable classes](#serializable), and
[annotations](#annotations), which are all discussed in their respective
examples.

### All possible applications in the input jars {: #applications}

These options shrink, optimize, and obfuscate all public applications in
`in.jar`:

    -injars      in.jar
    -outjars     out.jar
    -libraryjars <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)
    -printseeds

    -keepclasseswithmembers public class * {
        public static void main(java.lang.String[]);
    }

Note the use of [`-keepclasseswithmembers`](usage.md#keepclasseswithmembers).
We don't want to preserve all classes, just all classes that have main
methods, _and_ those methods.

The [`-printseeds`](usage.md#printseeds) option prints out which classes
exactly will be preserved, so we know for sure we're getting what we want.

If applicable, you should add options for processing [native
methods](#native), [callback methods](#callback),
[enumerations](#enumerations), [serializable classes](#serializable), [bean
classes](#beans), [annotations](#annotations), and [resource
files](#resourcefiles).

### All possible applets in the input jars {: #applets}

These options shrink, optimize, and obfuscate all public applets in `in.jar`:

    -injars      in.jar
    -outjars     out.jar
    -libraryjars <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)
    -libraryjars <java.home>/jmods/java.desktop.jmod(!**.jar;!module-info.class)
    -printseeds

    -keep public class * extends java.applet.Applet

We're simply keeping all classes that extend the `Applet` class.

Again, the [`-printseeds`](usage.md#printseeds) option prints out which
applets exactly will be preserved.

If applicable, you should add options for processing [native
methods](#native), [callback methods](#callback),
[enumerations](#enumerations), [serializable classes](#serializable), [bean
classes](#beans), [annotations](#annotations), and [resource
files](#resourcefiles).

### All possible midlets in the input jars {: #midlets}

These options shrink, optimize, obfuscate, and preverify all public midlets in
`in.jar`:

    -injars      in.jar
    -outjars     out.jar
    -libraryjars /usr/local/java/wtk2.5.2/lib/midpapi20.jar
    -libraryjars /usr/local/java/wtk2.5.2/lib/cldcapi11.jar
    -overloadaggressively
    -repackageclasses ''
    -allowaccessmodification
    -microedition
    -printseeds

    -keep public class * extends javax.microedition.midlet.MIDlet

We're simply keeping all classes that extend the `MIDlet` class.

The [`-microedition`](usage.md#microedition) option makes sure the class files
are preverified for Java Micro Edition, producing compact `StackMap`
attributes. It is no longer necessary to run an external preverifier.

Be careful if you do use the external `preverify` tool on a platform with a
case-insensitive filing system, such as Windows. Because this tool unpacks
your processed jars, you should then use ProGuard's
[`-dontusemixedcaseclassnames`](usage.md#dontusemixedcaseclassnames) option.

The [`-printseeds`](usage.md#printseeds) option prints out which midlets
exactly will be preserved.

If applicable, you should add options for processing [native methods](#native)
and [resource files](#resourcefiles).

Note that you will still have to adapt the midlet jar size in the
corresponding jad file; ProGuard doesn't do that for you.

### All possible Java Card applets in the input jars {: #jcapplets}

These options shrink, optimize, and obfuscate all public Java Card
applets in `in.jar`:

    -injars      in.jar
    -outjars     out.jar
    -libraryjars /usr/local/java/javacard2.2.2/lib/api.jar
    -dontwarn    java.lang.Class
    -overloadaggressively
    -repackageclasses ''
    -allowaccessmodification
    -printseeds

    -keep public class * implements javacard.framework.Applet

We're simply keeping all classes that implement the `Applet` interface.

The [`-printseeds`](usage.md#printseeds) option prints out which applets
exactly will be preserved.

### All possible xlets in the input jars {: #xlets}

These options shrink, optimize, and obfuscate all public xlets in `in.jar`:

    -injars      in.jar
    -outjars     out.jar
    -libraryjars /usr/local/java/jtv1.1/javatv.jar
    -libraryjars /usr/local/java/cdc1.1/lib/cdc.jar
    -libraryjars /usr/local/java/cdc1.1/lib/btclasses.zip
    -overloadaggressively
    -repackageclasses ''
    -allowaccessmodification
    -printseeds

    -keep public class * implements javax.tv.xlet.Xlet

We're simply keeping all classes that implement the `Xlet` interface.

The [`-printseeds`](usage.md#printseeds) option prints out which xlets exactly
will be preserved.

### All possible servlets in the input jars {: #servlets}

These options shrink, optimize, and obfuscate all public servlets in `in.jar`:

    -injars      in.jar
    -outjars     out.jar
    -libraryjars <java.home>/lib/rt.jar
    -libraryjars /usr/local/java/servlet/servlet.jar
    -printseeds

    -keep public class * implements javax.servlet.Servlet

Keeping all servlets is very similar to keeping all applets. The servlet API
is not part of the standard run-time jar, so we're specifying it as a library.
Don't forget to use the right path name.

We're then keeping all classes that implement the `Servlet` interface. We're
using the `implements` keyword because it looks more familiar in this context,
but it is equivalent to `extends`, as far as ProGuard is concerned.

The [`-printseeds`](usage.md#printseeds) option prints out which
servlets exactly will be preserved.

If applicable, you should add options for processing [native
methods](#native), [callback methods](#callback),
[enumerations](#enumerations), [serializable classes](#serializable), [bean
classes](#beans), [annotations](#annotations), and [resource
files](#resourcefiles).

### Scala applications with the Scala runtime {: #scala}

These options shrink, optimize, and obfuscate all public Scala applications in
`in.jar`:

    -injars      in.jar
    -injars      /usr/local/java/scala-2.9.1/lib/scala-library.jar
    -outjars     out.jar
    -libraryjars <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)

    -dontwarn scala.**

    -keepclasseswithmembers public class * {
        public static void main(java.lang.String[]);
    }

    -keep class * implements org.xml.sax.EntityResolver

    -keepclassmembers class * {
        ** MODULE$;
    }

    -keepclassmembernames class scala.concurrent.forkjoin.ForkJoinPool {
        long eventCount;
        int  workerCounts;
        int  runControl;
        scala.concurrent.forkjoin.ForkJoinPool$WaitQueueNode syncStack;
        scala.concurrent.forkjoin.ForkJoinPool$WaitQueueNode spareStack;
    }

    -keepclassmembernames class scala.concurrent.forkjoin.ForkJoinWorkerThread {
        int base;
        int sp;
        int runState;
    }

    -keepclassmembernames class scala.concurrent.forkjoin.ForkJoinTask {
        int status;
    }

    -keepclassmembernames class scala.concurrent.forkjoin.LinkedTransferQueue {
        scala.concurrent.forkjoin.LinkedTransferQueue$PaddedAtomicReference head;
        scala.concurrent.forkjoin.LinkedTransferQueue$PaddedAtomicReference tail;
        scala.concurrent.forkjoin.LinkedTransferQueue$PaddedAtomicReference cleanMe;
    }

The configuration is essentially the same as for [processing
applications](#applications), because Scala is compiled to ordinary Java
bytecode. However, the example processes the Scala runtime library as well.
The processed jar can be an order of magnitude smaller and a few times faster
than the original code (for the Scala code examples, for instance).

The [`-dontwarn`](usage.md#dontwarn) option tells ProGuard not to complain
about some artefacts in the Scala runtime, the way it is compiled by the
`scalac` compiler (at least in Scala 2.9.1 and older). Note that this option
should always be used with care.

The additional [`-keep`](usage.md#keepoverview) options make sure that some
classes and some fields that are accessed by means of introspection are not
removed or renamed.

If applicable, you should add options for processing [native
methods](#native), [callback methods](#callback),
[enumerations](#enumerations), [serializable classes](#serializable), [bean
classes](#beans), [annotations](#annotations), and [resource
files](#resourcefiles).

## Processing common code constructs {: #commonconstructs}

### Processing native methods {: #native}

If your application, applet, servlet, library, etc., contains native methods,
you'll want to preserve their names and their classes' names, so they can
still be linked to the native library. The following additional option will
ensure that:

    -keepclasseswithmembernames,includedescriptorclasses class * {
        native <methods>;
    }

Note the use of
[`-keepclasseswithmembernames`](usage.md#keepclasseswithmembernames). We don't
want to preserve all classes or all native methods; we just want to keep the
relevant names from being obfuscated. The modifier
[includedescriptorclasses](usage.md#includedescriptorclasses) additionally
makes sure that the return types and parameter types aren't renamed either, so
the entire signatures remain compatible with the native libraries.

ProGuard doesn't look at your native code, so it won't automatically preserve
the classes or class members that are invoked by the native code. These are
entry points, which you'll have to specify explicitly. [Callback
methods](callback) are discussed below as a typical example.

### Processing callback methods {: #callback}

If your application, applet, servlet, library, etc., contains callback
methods, which are called from external code (native code, scripts,...),
you'll want to preserve them, and probably their classes too. They are just
entry points to your code, much like, say, the main method of an application.
If they aren't preserved by other [`-keep`](usage.md#keep) options, something
like the following option will keep the callback class and method:

    -keep class com.example.MyCallbackClass {
        void myCallbackMethod(java.lang.String);
    }

This will preserve the given class and method from being removed or renamed.

### Processing enumeration classes {: #enumerations}

If your application, applet, servlet, library, etc., contains enumeration
classes, you'll have to preserve some special methods. Enumerations were
introduced in Java 5. The java compiler translates enumerations into classes
with a special structure. Notably, the classes contain implementations of some
static methods that the run-time environment accesses by introspection (Isn't
that just grand? Introspection is the self-modifying code of a new
generation). You have to specify these explicitly, to make sure they aren't
removed or obfuscated:

    -keepclassmembers,allowoptimization enum * {
        public static **[] values();
        public static ** valueOf(java.lang.String);
    }

### Processing serializable classes {: #serializable}

More complex applications, applets, servlets, libraries, etc., may
contain classes that are serialized. Depending on the way in which they
are used, they may require special attention:

- Often, serialization is simply a means of transporting data, without
  long-term storage. Classes that are shrunk and obfuscated should then
  continue to function fine with the following additional options:

        -keepclassmembers class * implements java.io.Serializable {
            private static final java.io.ObjectStreamField[] serialPersistentFields;
            private void writeObject(java.io.ObjectOutputStream);
            private void readObject(java.io.ObjectInputStream);
            java.lang.Object writeReplace();
            java.lang.Object readResolve();
        }

    The [`-keepclassmembers`](usage.md#keepclassmembers) option makes sure
    that any serialization methods are kept. By using this option instead of
    the basic `-keep` option, we're not forcing preservation of *all*
    serializable classes, just preservation of the listed members of classes
    that are actually used.

- Sometimes, the serialized data are stored, and read back later into newer
  versions of the serializable classes. One then has to take care the classes
  remain compatible with their unprocessed versions and with future processed
  versions. In such cases, the relevant classes will most likely have
  `serialVersionUID` fields. The following options should then be sufficient
  to ensure compatibility over time:

        -keepnames class * implements java.io.Serializable

        -keepclassmembers class * implements java.io.Serializable {
            static final long serialVersionUID;
            private static final java.io.ObjectStreamField[] serialPersistentFields;
            !static !transient <fields>;
            private void writeObject(java.io.ObjectOutputStream);
            private void readObject(java.io.ObjectInputStream);
            java.lang.Object writeReplace();
            java.lang.Object readResolve();
        }

    The `serialVersionUID` and `serialPersistentFields` lines makes sure those
    fields are preserved, if they are present. The `<fields>` line preserves
    all non-static, non-transient fields, with their original names. The
    introspection of the serialization process and the de-serialization
    process will then find consistent names.

- Occasionally, the serialized data have to remain compatible, but the classes
  involved lack `serialVersionUID` fields. I imagine the original code will
  then be hard to maintain, since the serial version UID is then computed from
  a list of features the serializable class. Changing the class ever so
  slightly may change the computed serial version UID. The list of features is
  specified in the section on [Stream Unique
  Identifiers](http://docs.oracle.com/javase/8/docs/platform/serialization/spec/class.html#a4100)
  of Sun's [Java Object Serialization
  Specification](http://docs.oracle.com/javase/8/docs/platform/serialization/spec/serialTOC.html).
  The following directives should at least partially ensure compatibility with
  the original classes:

        -keepnames class * implements java.io.Serializable

        -keepclassmembers class * implements java.io.Serializable {
            static final long serialVersionUID;
            private static final java.io.ObjectStreamField[] serialPersistentFields;
            !static !transient <fields>;
            !private <fields>;
            !private <methods>;
            private void writeObject(java.io.ObjectOutputStream);
            private void readObject(java.io.ObjectInputStream);
            java.lang.Object writeReplace();
            java.lang.Object readResolve();
        }

    The new options force preservation of the elements involved in the UID
    computation. In addition, the user will have to manually specify all
    interfaces of the serializable classes (using something like "`-keep
    interface MyInterface`"), since these names are also used when computing
    the UID. A fast but sub-optimal alternative would be simply keeping all
    interfaces with "`-keep interface *`".

- In the rare event that you are serializing lambda expressions in Java 8 or
  higher, you need to preserve some methods and adapt the hard-coded names of
  the classes in which they occur:

        -keepclassmembers class * {
            private static synthetic java.lang.Object $deserializeLambda$(java.lang.invoke.SerializedLambda);
        }

        -keepclassmembernames class * {
            private static synthetic *** lambda$*(...);
        }

        -adaptclassstrings com.example.Test

    This should satisfy the reflection in the deserialization code of the Java
    run-time.

Note that the above options may preserve more classes and class members than
strictly necessary. For instance, a large number of classes may implement the
`Serialization` interface, yet only a small number may actually ever be
serialized. Knowing your application and tuning the configuration often
produces more compact results.

### Processing bean classes {: #beans}

If your application, applet, servlet, library, etc., makes extensive use of
introspection on bean classes to find bean editor classes, or getter and
setter methods, then configuration may become painful. There's not much else
you can do than making sure the bean class names, or the getter and setter
names don't change. For instance:

    -keep public class com.example.MyBean {
        public void setMyProperty(int);
        public int getMyProperty();
    }

    -keep public class com.example.MyBeanEditor

If there are too many elements to list explicitly, wildcards in class
names and method signatures might be helpful. This example preserves all
possible setters and getters in classes in the package `mybeans`:

    -keep class mybeans.** {
        void set*(***);
        void set*(int, ***);

        boolean is*();
        boolean is*(int);

        *** get*();
        *** get*(int);
    }

The '`***`' wildcard matches any type (primitive or non-primitive, array
or non-array). The methods with the '`int`' arguments matches properties
that are lists.

### Processing annotations {: #annotations}

If your application, applet, servlet, library, etc., uses annotations, you may
want to preserve them in the processed output. Annotations are represented by
attributes that have no direct effect on the execution of the code. However,
their values can be retrieved through introspection, allowing developers to
adapt the execution behavior accordingly. By default, ProGuard treats
annotation attributes as optional, and removes them in the obfuscation step.
If they are required, you'll have to specify this explicitly:

    -keepattributes *Annotation*

For brevity, we're specifying a wildcarded attribute name, which will match
`RuntimeVisibleAnnotations`, `RuntimeInvisibleAnnotations`,
`RuntimeVisibleParameterAnnotations`, `RuntimeInvisibleParameterAnnotations`,
and `AnnotationDefault`. Depending on the purpose of the processed code, you
could refine this selection, for instance not keeping the run-time invisible
annotations (which are only used at compile-time).

Some code may make further use of introspection to figure out the enclosing
methods of anonymous inner classes. In that case, the corresponding attribute
has to be preserved as well:

    -keepattributes EnclosingMethod

### Processing database drivers {: #database}

Database drivers are implementations of the `Driver` interface. Since they are
often created dynamically, you may want to preserve any implementations that
you are processing as entry points:

    -keep class * implements java.sql.Driver

This option also gets rid of the note that ProGuard prints out about
`(java.sql.Driver)Class.forName` constructs, if you are instantiating a driver
in your code (without necessarily implementing any drivers yourself).

### Processing ComponentUI classes {: #componentui}

Swing UI look and feels are implemented as extensions of the `ComponentUI`
class. For some reason, these have to contain a static method `createUI`,
which the Swing API invokes using introspection. You should therefore always
preserve the method as an entry point, for instance like this:

    -keep class * extends javax.swing.plaf.ComponentUI {
        public static javax.swing.plaf.ComponentUI createUI(javax.swing.JComponent);
    }

This option also keeps the classes themselves.

## Processing common libraries {: #commonlibraries}

### Processing RMI code {: #rmi}

Reportedly, the easiest way to handle RMI code is to process the code with
ProGuard first and then invoke the `rmic` tool. If that is not possible, you
may want to try something like this:

    -keepattributes Exceptions

    -keep interface * extends java.rmi.Remote {
        <methods>;
    }

    -keep class * implements java.rmi.Remote {
        <init>(java.rmi.activation.ActivationID, java.rmi.MarshalledObject); {: #activation}
    }

The first [`-keep`](usage.md#keep) option keeps all your Remote interfaces and
their methods. The second one keeps all the implementations, along with their
particular RMI constructors, if any.

The `Exceptions` attribute has to be kept too, because the RMI handling code
performs introspection to check whether the method signatures are compatible.

### Optimizing Gson code {: #gson}

ProGuard [optimizes Gson code](optimizations.md#gson), by detecting which
domain classes are serialized using the Gson library, and then replacing the
reflection-based implementation by more efficient hard-coded serialization.

The GSON optimization is enabled by default and doesn't require any additional
configuration. If you've disabled optimization, the GSON library still relies
on reflection on the fields of the classes that it serializes. You then need
to preserve the parameterless constructor and the serialized fields from being
removed, optimized, or obfuscated. For example:

    -keepclassmembers class com.example.SerializedClass {
        <fields>;
        <init>();
    }

While creating the configuration, you can specify the option
[`-addconfigurationdebugging`](usage.md#addconfigurationdebugging), to get
feedback on the necessary settings at run-time.

Alternatively, you can make sure the fields are explicitly annotated with
`@SerializedName`, so the names of the fields can be obfuscated. You can
then keep all of them at the same time with:

    -keepclasseswithmembers,allowobfuscation,includedescriptorclasses class * {
        @com.google.gson.annotations.SerializedName <fields>;
    }

    -keepclassmembers enum * {
        @com.google.gson.annotations.SerializedName <fields>;
    }

### Processing dependency injection {: #injection}

If your application is using JEE-style dependency injection, the application
container will automatically assign instances of resource classes to fields
and methods that are annotated with `@Resource`. The container applies
introspection, even accessing private class members directly. It typically
constructs a resource name based on the type name and the class member name.
We then have to avoid that such class members are removed or renamed:

    -keepclassmembers class * {
        @javax.annotation.Resource *;
    }

The Spring framework has another similar annotation `@Autowired`:

    -keepclassmembers class * {
        @org.springframework.beans.factory.annotation.Autowired *;
    }

### Processing Dagger code {: #dagger}

Your Android application may be using the Dagger library for its dependency
injection.

**Dagger 1** relies heavily on reflection, so you may need some additional
configuration to make sure it continues to work. DexGuard's default
configuration already keeps some required classes:

    -keepclassmembers,allowobfuscation class * {
        @dagger.** *;
    }

    -keep class **$$ModuleAdapter
    -keep class **$$InjectAdapter
    -keep class **$$StaticInjection

    -if   class **$$ModuleAdapter
    -keep class <1>

    -if   class **$$InjectAdapter
    -keep class <1>

    -if   class **$$StaticInjection
    -keep class <1>

    -keepnames class dagger.Lazy

That way, Dagger can combine the corresponding pairs of classes, based on
their names.

Furthermore, if your code injects dependencies into some given classes with an
annotation like `@Module(injects = { SomeClass.class }, ...)`, you need to
preserve the specified names as well:

    -keep class com.example.SomeClass

**Dagger 2** no longer relies on reflection. You don't need to preserve any
classes there.

### Processing Butterknife code {: #butterknife}

If your Android application includes Butterknife to inject views, you also
need a few lines of configuration, since Butterknife relies on reflection to
tie together the code at runtime:

    -keep @interface butterknife.*

    -keepclasseswithmembers class * {
        @butterknife.* <fields>;
    }

    -keepclasseswithmembers class * {
        @butterknife.* <methods>;
    }

    -keepclasseswithmembers class * {
        @butterknife.On* <methods>;
    }

    -keep class **$$ViewInjector {
        public static void inject(...);
        public static void reset(...);
    }

    -keep class **$$ViewBinder {
        public static void bind(...);
        public static void unbind(...);
    }

    -if   class **$$ViewBinder
    -keep class <1>

    -keep class **_ViewBinding {
        <init>(<1>, android.view.View);
    }

    -if   class **_ViewBinding
    -keep class <1>

These settings preserve the Butterknife annotations, the annotated fields and
methods, and the generated classes and methods that Butterknife accesses by
reflection.

## Further processing possibilities {: #furtherpossibilities}

### Processing resource files {: #resourcefiles}

If your application, applet, servlet, library, etc., contains resource files,
it may be necessary to adapt their names and/or their contents when the
application is obfuscated. The following two options can achieve this
automatically:

    -adaptresourcefilenames    **.properties,**.gif,**.jpg
    -adaptresourcefilecontents **.properties,META-INF/MANIFEST.MF

The [`-adaptresourcefilenames`](usage.md#adaptresourcefilenames) option in
this case renames properties files and image files in the processed output,
based on the obfuscated names of their corresponding class files (if any). The
[`-adaptresourcefilecontents`](usage.md#adaptresourcefilecontents) option
looks for class names in properties files and in the manifest file, and
replaces these names by the obfuscated names (if any). You'll probably want to
adapt the filters to suit your application.

### Processing manifest files {: #manifestfiles}

As illustrated in the previous section, manifest files can be treated like
ordinary resource files. ProGuard can adapt obfuscated class names in the
files, but it won't make any other changes. If you want anything else, you
should apply an external tool. For instance, if a manifest file contains
signing information, you should sign the jar again after it has been
processed.

If you're merging several input jars into a single output jar, you'll have to
pick one, typically by specifying [filters](usage.md#filters):

    -injars  in1.jar
    -injars  in2.jar(!META-INF/MANIFEST.MF)
    -injars  in3.jar(!META-INF/MANIFEST.MF)
    -outjars out.jar

The filters will let ProGuard copy the manifest file from the first jar and
ignore any manifest files in the second and third input jars. Note that
ProGuard will leave the order of the files in the jars unchanged; manifest
files are not necessarily put first.

### Producing useful obfuscated stack traces {: #stacktrace}

These options let obfuscated applications or libraries produce stack traces
that can still be deciphered later on:

    -printmapping out.map

    -renamesourcefileattribute SourceFile
    -keepattributes SourceFile,LineNumberTable

We're keeping all source file attributes, but we're replacing their values by
the string "SourceFile". We could use any string. This string is already
present in all class files, so it doesn't take up any extra space. If you're
working with J++, you'll want to keep the "SourceDir" attribute as well.

We're also keeping the line number tables of all methods.

Whenever both of these attributes are present, the Java run-time environment
will include line number information when printing out exception stack traces.

The information will only be useful if we can map the obfuscated names back to
their original names, so we're saving the mapping to a file `out.map`. The
information can then be used by the [ReTrace](retrace/index.html) tool to
restore the original stack trace.

### Obfuscating package names {: #repackaging}

Package names can be obfuscated in various ways, with increasing levels of
obfuscation and compactness. For example, consider the following classes:

    mycompany.myapplication.MyMain
    mycompany.myapplication.Foo
    mycompany.myapplication.Bar
    mycompany.myapplication.extra.FirstExtra
    mycompany.myapplication.extra.SecondExtra
    mycompany.util.FirstUtil
    mycompany.util.SecondUtil

Let's assume the class name `mycompany.myapplication.MyMain` is the main
application class that is kept by the configuration. All other class names can
be obfuscated.

By default, packages that contain classes that can't be renamed aren't renamed
either, and the package hierarchy is preserved. This results in obfuscated
class names like these:

    mycompany.myapplication.MyMain
    mycompany.myapplication.a
    mycompany.myapplication.b
    mycompany.myapplication.a.a
    mycompany.myapplication.a.b
    mycompany.a.a
    mycompany.a.b

The [`-flattenpackagehierarchy`](usage.md#flattenpackagehierarchy) option
obfuscates the package names further, by flattening the package hierarchy of
obfuscated packages:

    -flattenpackagehierarchy 'myobfuscated'

The obfuscated class names then look as follows:

    mycompany.myapplication.MyMain
    mycompany.myapplication.a
    mycompany.myapplication.b
    myobfuscated.a.a
    myobfuscated.a.b
    myobfuscated.b.a
    myobfuscated.b.b

Alternatively, the [`-repackageclasses`](usage.md#repackageclasses) option
obfuscates the entire packaging, by combining obfuscated classes into a single
package:

    -repackageclasses 'myobfuscated'

The obfuscated class names then look as follows:

    mycompany.myapplication.MyMain
    mycompany.myapplication.a
    mycompany.myapplication.b
    myobfuscated.a
    myobfuscated.b
    myobfuscated.c
    myobfuscated.d

Additionally specifying the
[`-allowaccessmodification`](usage.md#allowaccessmodification) option allows
access permissions of classes and class members to be broadened, opening up
the opportunity to repackage all obfuscated classes:

    -repackageclasses 'myobfuscated'
    -allowaccessmodification

The obfuscated class names then look as follows:

    mycompany.myapplication.MyMain
    myobfuscated.a
    myobfuscated.b
    myobfuscated.c
    myobfuscated.d
    myobfuscated.e
    myobfuscated.f

The specified target package can always be the root package. For
instance:

    -repackageclasses ''
    -allowaccessmodification

The obfuscated class names are then the shortest possible names:

    mycompany.myapplication.MyMain
    a
    b
    c
    d
    e
    f

Note that not all levels of obfuscation of package names may be acceptable for
all code. Notably, you may have to take into account that your application may
contain [resource files](#resourcefiles) that have to be adapted.

### Removing logging code {: #logging}

You can let ProGuard remove logging code. The trick is to specify that the
logging methods don't have side-effects — even though they actually do, since
they write to the console or to a log file. ProGuard will take your word for
it and remove the invocations (in the optimization step) and if possible the
logging classes and methods themselves (in the shrinking step).

For example, this configuration removes invocations of the Android
logging methods:

    -assumenosideeffects class android.util.Log {
        public static boolean isLoggable(java.lang.String, int);
        public static int v(...);
        public static int i(...);
        public static int w(...);
        public static int d(...);
        public static int e(...);
    }

The wildcards are a shortcut to match all versions of the methods. Be careful
not to use a `*` wildcard to match all methods, because it would also match
methods like `wait()`, higher up the hierarchy. Removing those invocations
will generally break your code.

Note that you generally can't remove logging code that uses
`System.out.println`, since you would be removing all invocations of
`java.io.PrintStream#println`, which could break your application. You can
work around it by creating your own logging methods and let ProGuard remove
those.

Logging statements often contain implicit calls that perform string
concatenation. They no longer serve a purpose after the logging calls have
been removed. You can let ProGuard clean up such constructs as well by
providing additional hints:

    -assumenoexternalsideeffects class java.lang.StringBuilder {
        public java.lang.StringBuilder();
        public java.lang.StringBuilder(int);
        public java.lang.StringBuilder(java.lang.String);
        public java.lang.StringBuilder append(java.lang.Object);
        public java.lang.StringBuilder append(java.lang.String);
        public java.lang.StringBuilder append(java.lang.StringBuffer);
        public java.lang.StringBuilder append(char[]);
        public java.lang.StringBuilder append(char[], int, int);
        public java.lang.StringBuilder append(boolean);
        public java.lang.StringBuilder append(char);
        public java.lang.StringBuilder append(int);
        public java.lang.StringBuilder append(long);
        public java.lang.StringBuilder append(float);
        public java.lang.StringBuilder append(double);
        public java.lang.String toString();
    }

    -assumenoexternalreturnvalues public final class java.lang.StringBuilder {
        public java.lang.StringBuilder append(java.lang.Object);
        public java.lang.StringBuilder append(java.lang.String);
        public java.lang.StringBuilder append(java.lang.StringBuffer);
        public java.lang.StringBuilder append(char[]);
        public java.lang.StringBuilder append(char[], int, int);
        public java.lang.StringBuilder append(boolean);
        public java.lang.StringBuilder append(char);
        public java.lang.StringBuilder append(int);
        public java.lang.StringBuilder append(long);
        public java.lang.StringBuilder append(float);
        public java.lang.StringBuilder append(double);
    }

Be careful specifying your own assumptions, since they can easily break
your code.

### Optimizing for Android SDK versions {: #androidsdk}

You can let ProGuard optimize the code for the range of Android versions that
you intend to support — the range between the minimum SDK version and the
maximum SDK version in your Android manifest. It then removes all code for SDK
versions that are not relevant, for example in the various Android support
libraries.

For example, if the minimum SDK version in your Android manifest is 19, you
can optimize the code accordingly:

    -assumevalues class android.os.Build$VERSION {
        int SDK_INT = 19..2147483647;
    }

You can also specify return values for methods. The "`=`" keyword and the
"`return`" keyword are equivalent. Be careful specifying assumptions, since
they can easily break your code.

### Restructuring the output archives {: #restructuring}

In simple applications, all output classes and resources files are merged into
a single jar. For example:

    -injars  classes
    -injars  in1.jar
    -injars  in2.jar
    -injars  in3.jar
    -outjars out.jar

This configuration merges the processed versions of the files in the `classes`
directory and the three jars into a single output jar `out.jar`.

If you want to preserve the structure of your input jars (and/or apks, aars,
wars, ears, jmods, zips, or directories), you can specify an output directory
(or an apk, an aar, a war, an ear, a jmod, or a zip). For example:

    -injars  in1.jar
    -injars  in2.jar
    -injars  in3.jar
    -outjars out

The input jars will then be reconstructed in the directory `out`, with their
original names.

You can also combine archives into higher level archives. For example:

    -injars  in1.jar
    -injars  in2.jar
    -injars  in3.jar
    -outjars out.war

The other way around, you can flatten the archives inside higher level
archives into simple archives:

    -injars  in.war
    -outjars out.jar

This configuration puts the processed contents of all jars inside `in.war`
(plus any other contents of `in.war`) into `out.jar`.

If you want to combine input jars (and/or apks, aars, wars, ears, jmods, zips,
or directories) into output jars (and/or apks, aars, wars, ears, jmods, zips,
or directories), you can group the [`-injars`](usage.md#injars) and
[`-outjars`](usage.md#outjars) options. For example:

    -injars base_in1.jar
    -injars base_in2.jar
    -injars base_in3.jar
    -outjars base_out.jar

    -injars  extra_in.jar
    -outjars extra_out.jar

This configuration puts the processed results of all `base_in*.jar` jars into
`base_out.jar`, and the processed results of the `extra_in.jar` into
`extra_out.jar`. Note that only the order of the options matters; the
additional whitespace is just for clarity.

This grouping, archiving, and flattening can be arbitrarily complex. ProGuard
always tries to package output archives in a sensible way, reconstructing the
input entries as much as required.

### Filtering the input and the output {: #filtering}

If you want even greater control, you can add [filters](usage.md#filters) to
the input and the output, filtering out apks, jars, aars, wars, ears, jmods,
zips, and/or ordinary files. For example, if you want to disregard certain
files from an input jar:

    -injars  in.jar(!images/**)
    -outjars out.jar

This configuration removes any files in the `images` directory and its
subdirectories.

Such filters can be convenient for avoiding warnings about duplicate files in
the output. For example, only keeping the manifest file from a first input
jar:

    -injars  in1.jar
    -injars  in2.jar(!META-INF/MANIFEST.MF)
    -injars  in3.jar(!META-INF/MANIFEST.MF)
    -outjars out.jar

Another useful application is ignoring unwanted files from the runtime library
module:

    -libraryjars <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)

The filter makes ProGuard disregard redundant jars inside the module, and
module info classes that would only cause conflicts with duplicate names.

It is also possible to filter the jars (and/or apks, aabs, aars, wars, ears,
jmods, zips) themselves, based on their names. For example:

    -injars  in(**/acme_*.jar;)
    -outjars out.jar

Note the semi-colon in the filter; the filter in front of it applies to jar
names. In this case, only `acme_*.jar` jars are read from the directory `in`
and its subdirectories. Filters for war names, ear names, and zip names can be
prefixed with additional semi-colons. All types of filters can be combined.
They are orthogonal.

On the other hand, you can also filter the output, in order to control what
content goes where. For example:

    -injars  in.jar
    -outjars code_out.jar(**.class)
    -outjars resources_out.jar

This configuration splits the processed output, sending `**.class` files to
`code_out.jar`, and all remaining files to `resources_out.jar`.

Again, the filtering can be arbitrarily complex, especially when combined with
grouping input and output.

### Processing multiple applications at once {: #multiple}

You can process several dependent or independent applications (or applets,
midlets,...) in one go, in order to save time and effort. ProGuard's input and
output handling offers various ways to keep the output nicely structured.

The easiest way is to specify your input jars (and/or wars, ears, zips, and
directories) and a single output directory. ProGuard will then reconstruct the
input in this directory, using the original jar names. For example, showing
just the input and output options:

    -injars  application1.jar
    -injars  application2.jar
    -injars  application3.jar
    -outjars processed_applications

After processing, the directory `processed_applications` will contain
processed versions of application jars, with their original names.

### Incremental obfuscation {: #incremental}

After having [processed an application](#application), e.g. ProGuard itself,
you can still incrementally add other pieces of code that depend on it, e.g.
the ProGuard GUI:

    -injars       proguardgui.jar
    -outjars      proguardgui_out.jar
    -injars       proguard.jar
    -outjars      proguard_out.jar
    -libraryjars  <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)
    -applymapping proguard.map

    -keep public class proguard.gui.ProGuardGUI {
        public static void main(java.lang.String[]);
    }

We're reading both unprocessed jars as input. Their processed contents will go
to the respective output jars. The [`-applymapping`](usage.md#applymapping)
option then makes sure the ProGuard part of the code gets the previously
produced obfuscation mapping. The final application will consist of the
obfuscated ProGuard jar and the additional obfuscated GUI jar.

The added code in this example is straightforward; it doesn't affect the
original code. The `proguard_out.jar` will be identical to the one produced in
the initial processing step. If you foresee adding more complex extensions to
your code, you should specify the options
[`-useuniqueclassmembernames`](usage.md#useuniqueclassmembernames),
[`-dontshrink`](usage.md#dontshrink), and
[`-dontoptimize`](usage.md#dontoptimize) *in the original processing step*.
These options ensure that the obfuscated base jar will always remain usable
without changes. You can then specify the base jar as a library jar:

    -injars       proguardgui.jar
    -outjars      proguardgui_out.jar
    -libraryjars  proguard.jar
    -libraryjars  <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)
    -applymapping proguard.map

    -keep public class proguard.gui.ProGuardGUI {
        public static void main(java.lang.String[]);
    }

## Other uses {: #otheruses}

### Preverifying class files for Java Micro Edition {: #microedition}

Even if you're not interested in shrinking, optimizing, and obfuscating your
midlets, as shown in the [midlets example](#midlets), you can still use
ProGuard to preverify the class files for Java Micro Edition. ProGuard
produces slightly more compact results than the traditional external
preverifier.

    -injars      in.jar
    -outjars     out.jar
    -libraryjars /usr/local/java/wtk2.5.2/lib/midpapi20.jar
    -libraryjars /usr/local/java/wtk2.5.2/lib/cldcapi11.jar

    -dontshrink
    -dontoptimize
    -dontobfuscate

    -microedition

We're not processing the input, just making sure the class files are
preverified by targeting them at Java Micro Edition with the
[`-microedition`](usage.md#microedition) option. Note that we don't need any
[`-keep`](usage.md#keep) options to specify entry points; all class files are
simply preverified.

### Upgrading old class files to Java 6 {: #upgrade}

The following options upgrade class files to Java 6, by updating their
internal version numbers and preverifying them. The class files can then be
loaded more efficiently by the Java 6 Virtual Machine.

    -injars      in.jar
    -outjars     out.jar
    -libraryjars <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)

    -dontshrink
    -dontoptimize
    -dontobfuscate

    -target 1.6

We're not processing the input, just retargeting the class files with the
[`-target`](usage.md#target) option. They will automatically be preverified
for Java 6 as a result. Note that we don't need any `-keep` options to specify
entry points; all class files are simply updated and preverified.

### Finding dead code {: #deadcode}

These options list unused classes, fields, and methods in the application
`com.example.MyApplication`:

    -injars      in.jar
    -libraryjars <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)

    -dontoptimize
    -dontobfuscate
    -dontpreverify
    -printusage

    -keep public class com.example.MyApplication {
        public static void main(java.lang.String[]);
    }

We're not specifying an output jar, just printing out some results. We're
saving some processing time by skipping the other processing steps.

The java compiler inlines primitive constants and String constants (`static
final` fields). ProGuard would therefore list such fields as not being used in
the class files that it analyzes, even if they *are* used in the source files.
We can add a [`-keepclassmembers`](usage.md#keepclassmembers) option that
keeps those fields a priori, in order to avoid having them listed:

    -keepclassmembers class * {
        static final %                *;
        static final java.lang.String *;
    }

### Printing out the internal structure of class files {: #structure}

These options print out the internal structure of all class files in the
input jar:

    -injars in.jar

    -dontshrink
    -dontoptimize
    -dontobfuscate
    -dontpreverify

    -dump

Note how we don't need to specify the Java run-time jar, because we're not
processing the input jar at all.

### Using annotations to configure ProGuard {: #annotated}

The traditional ProGuard configuration allows to keep a clean separation
between the code and the configuration for shrinking, optimization, and
obfuscation. However, it is also possible to define specific annotations, and
then annotate the code to configure the processing.

You can find a set of such predefined annotations in `lib/annotations.jar` in
the ProGuard distribution. The corresponding ProGuard configuration (or
meta-configuration, if you prefer) is specified in
`annotations/annotations.pro`. With these files, you can start annotating your
code. For instance, a java source file `Application.java` can be annotated as
follows:

    @KeepApplication
    public class Application {
      ....
    }

The ProGuard configuration file for the application can then be simplified by
leveraging these annotations:

    -injars      in.jar
    -outjars     out.jar
    -libraryjars <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)

    -include lib/annotations.pro

The annotations are effectively replacing the application-dependent `-keep`
options. You may still wish to add traditional [`-keep`](usage.md#keep)
options for processing [native methods](#native),
[enumerations](#enumerations), [serializable classes](#serializable), and
[annotations](#annotations).

The directory `examples/annotations` contains more examples that illustrate
some of the possibilities.
