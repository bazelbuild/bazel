#
# This ProGuard configuration file specifies how annotations can be used
# to configure the processing of other code.
# Usage:
#     java -jar proguard.jar @annotations.pro -libraryjars annotations.jar ...
#
# Note that the other input/output options still have to be specified.
# If you specify them in a separate file, you can simply include this file:
#     -include annotations.pro
#
# You can add any other options that are required. For instance, if you are
# processing a library, you can still include the options from library.pro.


# The annotations are defined in the accompanying jar. For now, we'll start
# with these. You can always define your own annotations, if necessary.
-libraryjars annotations.jar


# The following annotations can be specified with classes and with class
# members.

# @Keep specifies not to shrink, optimize, or obfuscate the annotated class
# or class member as an entry point.

-keep @proguard.annotation.Keep class *

-keepclassmembers class * {
    @proguard.annotation.Keep *;
}


# @KeepName specifies not to optimize or obfuscate the annotated class or
# class member as an entry point.

-keepnames @proguard.annotation.KeepName class *

-keepclassmembernames class * {
    @proguard.annotation.KeepName *;
}


# The following annotations can only be specified with classes.

# @KeepImplementations and @KeepPublicImplementations specify to keep all,
# resp. all public, implementations or extensions of the annotated class as
# entry points. Note the extension of the java-like syntax, adding annotations
# before the (wild-carded) interface name.

-keep        class * implements @proguard.annotation.KeepImplementations       *
-keep public class * implements @proguard.annotation.KeepPublicImplementations *

# @KeepApplication specifies to keep the annotated class as an application,
# together with its main method.

-keepclasseswithmembers @proguard.annotation.KeepApplication public class * {
    public static void main(java.lang.String[]);
}

# @KeepClassMembers, @KeepPublicClassMembers, and
# @KeepPublicProtectedClassMembers specify to keep all, all public, resp.
# all public or protected, class members of the annotated class from being
# shrunk, optimized, or obfuscated as entry points.

-keepclassmembers @proguard.annotation.KeepClassMembers class * {
    *;
}

-keepclassmembers @proguard.annotation.KeepPublicClassMembers class * {
    public *;
}

-keepclassmembers @proguard.annotation.KeepPublicProtectedClassMembers class * {
    public protected *;
}

# @KeepClassMemberNames, @KeepPublicClassMemberNames, and
# @KeepPublicProtectedClassMemberNames specify to keep all, all public, resp.
# all public or protected, class members of the annotated class from being
# optimized or obfuscated as entry points.

-keepclassmembernames @proguard.annotation.KeepClassMemberNames class * {
    *;
}

-keepclassmembernames @proguard.annotation.KeepPublicClassMemberNames class * {
    public *;
}

-keepclassmembernames @proguard.annotation.KeepPublicProtectedClassMemberNames class * {
    public protected *;
}

# @KeepGettersSetters and @KeepPublicGettersSetters specify to keep all, resp.
# all public, getters and setters of the annotated class from being shrunk,
# optimized, or obfuscated as entry points.

-keepclassmembers @proguard.annotation.KeepGettersSetters class * {
    void set*(***);
    void set*(int, ***);

    boolean is*();
    boolean is*(int);

    *** get*();
    *** get*(int);
}

-keepclassmembers @proguard.annotation.KeepPublicGettersSetters class * {
    public void set*(***);
    public void set*(int, ***);

    public boolean is*();
    public boolean is*(int);

    public *** get*();
    public *** get*(int);
}
