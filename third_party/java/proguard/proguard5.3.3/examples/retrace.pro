#
# This ProGuard configuration file illustrates how to process the ReTrace tool.
# Configuration files for typical applications will be very similar.
# Usage:
#     java -jar proguard.jar @retrace.pro
#

# Specify the input jars, output jars, and library jars.
# The input jars will be merged in a single output jar.
# We'll filter out the Ant and WTK classes.

-injars  ../lib/retrace.jar
-injars  ../lib/proguard.jar(!META-INF/MANIFEST.MF,
                             !proguard/ant/**,!proguard/wtk/**)
-outjars retrace_out.jar

-libraryjars <java.home>/lib/rt.jar

# If we wanted to reuse the previously obfuscated proguard_out.jar, we could
# perform incremental obfuscation based on its mapping file, and only keep the
# additional ReTrace files instead of all files.

#-applymapping proguard.map
#-outjars      retrace_out.jar(proguard/retrace/**)

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
