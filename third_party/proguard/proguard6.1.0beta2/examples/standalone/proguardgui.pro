#
# This ProGuard configuration file illustrates how to process the ProGuard GUI.
# Configuration files for typical applications will be very similar.
# Usage:
#     java -jar proguard.jar @proguardgui.pro
#

-verbose

# Specify the input jars, output jars, and library jars.
# The input jars will be merged in a single output jar.
# We'll filter out the Ant classes, Gradle classes, and WTK classes, keeping
# everything else.

-injars  ../../lib/proguardgui.jar
-injars  ../../lib/proguard.jar(!META-INF/**,!proguard/gradle/**,!proguard/ant/**,!proguard/wtk/**)
-injars  ../../lib/retrace.jar (!META-INF/**)
-outjars proguardgui_out.jar

# Before Java 9, the runtime classes were packaged in a single jar file.
#-libraryjars <java.home>/lib/rt.jar

# As of Java 9, the runtime classes are packaged in modular jmod files.
-libraryjars <java.home>/jmods/java.base.jmod   (!**.jar;!module-info.class)
-libraryjars <java.home>/jmods/java.desktop.jmod(!**.jar;!module-info.class)

# If we wanted to reuse the previously obfuscated proguard_out.jar, we could
# perform incremental obfuscation based on its mapping file, and only keep the
# additional GUI files instead of all files.

#-applymapping proguard.map
#-injars      ../../lib/proguardgui.jar
#-outjars     proguardgui_out.jar
#-libraryjars ../../lib/proguard.jar(!proguard/ant/**,!proguard/wtk/**)
#-libraryjars ../../lib/retrace.jar
#-libraryjars <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)
#-libraryjars <java.home>/jmods/java.desktop.jmod(!**.jar;!module-info.class)

# Don't print notes about reflection in injected code.

-dontnote proguard.configuration.ConfigurationLogger

# Allow methods with the same signature, except for the return type,
# to get the same obfuscation name.

-overloadaggressively

# Put all obfuscated classes into the nameless root package.

-repackageclasses ''

# Adapt the names of resource files, based on the corresponding obfuscated
# class names. Notably, in this case, the GUI resource properties file will
# have to be renamed.

-adaptresourcefilenames **.properties,**.gif,**.jpg

# The entry point: ProGuardGUI and its main method.

-keep public class proguard.gui.ProGuardGUI {
    public static void main(java.lang.String[]);
}
