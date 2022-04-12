#
# This ProGuard configuration file illustrates how to process the ReTrace tool.
# Configuration files for typical applications will be very similar.
# Usage:
#     java -jar proguard.jar @retrace.pro
#

-verbose

# Specify the input jars, output jars, and library jars.
# The input jars will be merged in a single output jar.
# We'll filter out the Ant and WTK classes.

-injars  ../../lib/retrace.jar
-injars  ../../lib/proguard.jar(!META-INF/MANIFEST.MF,!proguard/gui/**,!proguard/gradle/**,!proguard/ant/**,!proguard/wtk/**)
-outjars retrace_out.jar

# Before Java 9, the runtime classes were packaged in a single jar file.
#-libraryjars <java.home>/lib/rt.jar

# As of Java 9, the runtime classes are packaged in modular jmod files.
-libraryjars <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)
#-libraryjars <java.home>/jmods/.....

# If we wanted to reuse the previously obfuscated proguard_out.jar, we could
# perform incremental obfuscation based on its mapping file, and only keep the
# additional ReTrace files instead of all files.

#-applymapping proguard.map
#-outjars      retrace_out.jar(proguard/retrace/**)

# Don't print notes about reflection in injected code.

-dontnote proguard.configuration.ConfigurationLogger

# Don't print warnings about GSON dependencies.

-dontwarn com.google.gson.**

# Preserve injected GSON utility classes and their members.

-keep,allowobfuscation class proguard.optimize.gson._*
-keepclassmembers class proguard.optimize.gson._* {
    *;
}

# Obfuscate class strings of injected GSON utility classes.

-adaptclassstrings proguard.optimize.gson.**

# Allow methods with the same signature, except for the return type,
# to get the same obfuscation name.

-overloadaggressively

# Put all obfuscated classes into the nameless root package.

-repackageclasses ''

# Allow classes and class members to be made public.

-allowaccessmodification

# The entry point: ReTrace and its main method.

-keep public class proguard.retrace.ReTrace {
    public static void main(java.lang.String[]);
}
