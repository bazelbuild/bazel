#
# This ProGuard configuration file illustrates how to use annotations for
# specifying which classes and class members should be kept.
# Usage:
#     java -jar proguard.jar @examples.pro
#

# Specify the input, output, and library jars.
# This is assuming the code has been compiled in the examples directory.

-injars  examples(*.class)
-outjars out

# Before Java 9, the runtime classes were packaged in a single jar file.
#-libraryjars <java.home>/lib/rt.jar

# As of Java 9, the runtime classes are packaged in modular jmod files.
-libraryjars <java.home>/jmods/java.base.jmod(!**.jar;!module-info.class)
#-libraryjars <java.home>/jmods/.....

# Some important configuration is based on the annotations in the code.
# We have to specify what the annotations mean to ProGuard.

-include lib/annotations.pro

#
# We can then still add any other options that might be useful.
#

# Print out a list of what we're preserving.

-printseeds

# Preserve all annotations themselves.

-keepattributes *Annotation*

# Preserve all native method names and the names of their classes.

-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}

# Preserve the special static methods that are required in all enumeration
# classes.

-keepclassmembers,allowoptimization enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Explicitly preserve all serialization members. The Serializable interface
# is only a marker interface, so it wouldn't save them.
# You can comment this out if your application doesn't use serialization.
# If your code contains serializable classes that have to be backward 
# compatible, please refer to the manual.

-keepclassmembers class * implements java.io.Serializable {
    static final long serialVersionUID;
    static final java.io.ObjectStreamField[] serialPersistentFields;
    private void writeObject(java.io.ObjectOutputStream);
    private void readObject(java.io.ObjectInputStream);
    java.lang.Object writeReplace();
    java.lang.Object readResolve();
}
