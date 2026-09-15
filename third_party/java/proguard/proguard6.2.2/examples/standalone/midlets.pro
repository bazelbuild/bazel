#
# This ProGuard configuration file illustrates how to process J2ME midlets.
# Usage:
#     java -jar proguard.jar @midlets.pro
#

-verbose

# Specify the input jars, output jars, and library jars.

-injars  in.jar
-outjars out.jar

-libraryjars /usr/local/java/wtk2.5.2/lib/midpapi20.jar
-libraryjars /usr/local/java/wtk2.5.2/lib/cldcapi11.jar

# Preverify the code suitably for Java Micro Edition.

-microedition

# Allow methods with the same signature, except for the return type,
# to get the same obfuscation name.

-overloadaggressively

# Put all obfuscated classes into the nameless root package.

-repackageclasses ''

# Allow classes and class members to be made public.

-allowaccessmodification

# On Windows, you can't use mixed case class names,
# should you still want to use the preverify tool.
#
# -dontusemixedcaseclassnames

# Save the obfuscation mapping to a file, so you can de-obfuscate any stack
# traces later on.

-printmapping out.map

# You can keep a fixed source file attribute and all line number tables to
# get stack traces with line numbers.

#-renamesourcefileattribute SourceFile
#-keepattributes SourceFile,LineNumberTable

# You can print out the seeds that are matching the keep options below.

#-printseeds out.seeds

# Preserve all public midlets.

-keep public class * extends javax.microedition.midlet.MIDlet

# Preserve all native method names and the names of their classes.

-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}

# Your midlet may contain more items that need to be preserved; 
# typically classes that are dynamically created using Class.forName:

# -keep public class com.example.MyClass
# -keep public interface com.example.MyInterface
# -keep public class * implements com.example.MyInterface
