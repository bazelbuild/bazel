#
# This ProGuard configuration file illustrates how to process ProGuard itself.
# Configuration files for typical applications will be very similar.
# Usage:
#     java -jar proguard.jar @proguard.pro
#

-verbose

# Specify the input jars, output jars, and library jars.
# We'll filter out the Ant classes, Gradle classes, and WTK classes, keeping
# everything else.

-injars  ../../lib/proguard.jar(!proguard/ant/**,!proguard/gradle/**,!proguard/wtk/**)
-outjars proguard_out.jar

# Before Java 9, the runtime classes were packaged in a single jar file.
#-libraryjars <java.home>/lib/rt.jar

# As of Java 9, the runtime classes are packaged in modular jmod files.
-libraryjars <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)

# Write out an obfuscation mapping file, for de-obfuscating any stack traces
# later on, or for incremental obfuscation of extensions.

-printmapping proguard.map

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

# The entry point: ProGuard and its main method.

-keep public class proguard.ProGuard {
    public static void main(java.lang.String[]);
}

# If you want to preserve the Ant task as well, you'll have to specify the
# main ant.jar.

#-libraryjars /usr/local/java/ant/lib/ant.jar
#-adaptresourcefilecontents proguard/ant/task.properties
#
#-keep,allowobfuscation class proguard.ant.*
#-keepclassmembers public class proguard.ant.* {
#    <init>(org.apache.tools.ant.Project);
#    public void set*(***);
#    public void add*(***);
#}

# If you want to preserve the Gradle task, you'll have to specify the Gradle
# jars.

#-libraryjars /usr/local/java/gradle-4.2.1/lib/plugins/gradle-plugins-4.2.1.jar
#-libraryjars /usr/local/java/gradle-4.2.1/lib/gradle-base-services-4.2.1.jar
#-libraryjars /usr/local/java/gradle-4.2.1/lib/gradle-core-4.2.1.jar
#-libraryjars /usr/local/java/gradle-4.2.1/lib/groovy-all-2.4.12.jar

#-keep public class proguard.gradle.* {
#    public *;
#}

# If you want to preserve the WTK obfuscation plug-in, you'll have to specify
# the kenv.zip file.

#-libraryjars /usr/local/java/wtk2.5.2/wtklib/kenv.zip
#-keep public class proguard.wtk.ProGuardObfuscator
