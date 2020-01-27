-dontnote
-microedition
-mergeinterfacesaggressively
-overloadaggressively
-repackageclasses ''
-allowaccessmodification

# Keep all extensions of javax.microedition.midlet.MIDlet.
-keep public class * extends javax.microedition.midlet.MIDlet

# Keep all native class/method names.
-keepclasseswithmembernames class * {
    native <methods>;
}

# Remove all invocations of System methods without side effects
# whose return values are not used.
-assumenosideeffects public class java.lang.System {
    public static native long currentTimeMillis();
    static java.lang.Class getCallerClass();
    public static native int identityHashCode(java.lang.Object);
    public static java.lang.SecurityManager getSecurityManager();
    public static java.util.Properties getProperties();
    public static java.lang.String getProperty(java.lang.String);
    public static java.lang.String getenv(java.lang.String);
    public static native java.lang.String mapLibraryName(java.lang.String);
    public static java.lang.String getProperty(java.lang.String,java.lang.String);
}

# Remove all invocations of String methods without side effects
# whose return values are not used.
-assumenosideeffects public class java.lang.String {
    public java.lang.String();
    public java.lang.String(byte[]);
    public java.lang.String(byte[],int);
    public java.lang.String(byte[],int,int);
    public java.lang.String(byte[],int,int,int);
    public java.lang.String(byte[],int,int,java.lang.String);
    public java.lang.String(byte[],java.lang.String);
    public java.lang.String(char[]);
    public java.lang.String(char[],int,int);
    public java.lang.String(java.lang.String);
    public java.lang.String(java.lang.StringBuffer);
    public static java.lang.String copyValueOf(char[]);
    public static java.lang.String copyValueOf(char[],int,int);
    public static java.lang.String valueOf(boolean);
    public static java.lang.String valueOf(char);
    public static java.lang.String valueOf(char[]);
    public static java.lang.String valueOf(char[],int,int);
    public static java.lang.String valueOf(double);
    public static java.lang.String valueOf(float);
    public static java.lang.String valueOf(int);
    public static java.lang.String valueOf(java.lang.Object);
    public static java.lang.String valueOf(long);
    public boolean contentEquals(java.lang.StringBuffer);
    public boolean endsWith(java.lang.String);
    public boolean equalsIgnoreCase(java.lang.String);
    public boolean equals(java.lang.Object);
    public boolean matches(java.lang.String);
    public boolean regionMatches(boolean,int,java.lang.String,int,int);
    public boolean regionMatches(int,java.lang.String,int,int);
    public boolean startsWith(java.lang.String);
    public boolean startsWith(java.lang.String,int);
    public byte[] getBytes();
    public byte[] getBytes(java.lang.String);
    public char charAt(int);
    public char[] toCharArray();
    public int compareToIgnoreCase(java.lang.String);
    public int compareTo(java.lang.Object);
    public int compareTo(java.lang.String);
    public int hashCode();
    public int indexOf(int);
    public int indexOf(int,int);
    public int indexOf(java.lang.String);
    public int indexOf(java.lang.String,int);
    public int lastIndexOf(int);
    public int lastIndexOf(int,int);
    public int lastIndexOf(java.lang.String);
    public int lastIndexOf(java.lang.String,int);
    public int length();
    public java.lang.CharSequence subSequence(int,int);
    public java.lang.String concat(java.lang.String);
    public java.lang.String replaceAll(java.lang.String,java.lang.String);
    public java.lang.String replace(char,char);
    public java.lang.String replaceFirst(java.lang.String,java.lang.String);
    public java.lang.String[] split(java.lang.String);
    public java.lang.String[] split(java.lang.String,int);
    public java.lang.String substring(int);
    public java.lang.String substring(int,int);
    public java.lang.String toLowerCase();
    public java.lang.String toLowerCase(java.util.Locale);
    public java.lang.String toString();
    public java.lang.String toUpperCase();
    public java.lang.String toUpperCase(java.util.Locale);
    public java.lang.String trim();
}


# Remove all invocations of StringBuffer methods without side effects
# whose return values are not used.
-assumenosideeffects public class java.lang.StringBuffer {
    public java.lang.StringBuffer();
    public java.lang.StringBuffer(int);
    public java.lang.StringBuffer(java.lang.String);
    public java.lang.String toString();
    public char charAt(int);
    public int capacity();
    public int indexOf(java.lang.String,int);
    public int lastIndexOf(java.lang.String);
    public int lastIndexOf(java.lang.String,int);
    public int length();
    public java.lang.String substring(int);
    public java.lang.String substring(int,int);
}
