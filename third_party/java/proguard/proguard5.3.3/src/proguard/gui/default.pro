# The default configuration when starting up the GUI.

-libraryjars <java.home>/lib/rt.jar

# Keep - Applications. Keep all application classes, along with their 'main'
# methods.
-keepclasseswithmembers public class * {
    public static void main(java.lang.String[]);
}

# Also keep - Enumerations. Keep the special static methods that are required in
# enumeration classes.
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Also keep - Database drivers. Keep all implementations of java.sql.Driver.
-keep class * extends java.sql.Driver

# Also keep - Swing UI L&F. Keep all extensions of javax.swing.plaf.ComponentUI,
# along with the special 'createUI' method.
-keep class * extends javax.swing.plaf.ComponentUI {
    public static javax.swing.plaf.ComponentUI createUI(javax.swing.JComponent);
}

# Keep names - Native method names. Keep all native class/method names.
-keepclasseswithmembers,includedescriptorclasses,allowshrinking class * {
    native <methods>;
}

# Remove - System method calls. Remove all invocations of System
# methods without side effects whose return values are not used.
-assumenosideeffects public class java.lang.System {
    public static long currentTimeMillis();
    static java.lang.Class getCallerClass();
    public static int identityHashCode(java.lang.Object);
    public static java.lang.SecurityManager getSecurityManager();
    public static java.util.Properties getProperties();
    public static java.lang.String getProperty(java.lang.String);
    public static java.lang.String getenv(java.lang.String);
    public static java.lang.String mapLibraryName(java.lang.String);
    public static java.lang.String getProperty(java.lang.String,java.lang.String);
}

# Remove - Math method calls. Remove all invocations of Math
# methods without side effects whose return values are not used.
-assumenosideeffects public class java.lang.Math {
    public static double sin(double);
    public static double cos(double);
    public static double tan(double);
    public static double asin(double);
    public static double acos(double);
    public static double atan(double);
    public static double toRadians(double);
    public static double toDegrees(double);
    public static double exp(double);
    public static double log(double);
    public static double log10(double);
    public static double sqrt(double);
    public static double cbrt(double);
    public static double IEEEremainder(double,double);
    public static double ceil(double);
    public static double floor(double);
    public static double rint(double);
    public static double atan2(double,double);
    public static double pow(double,double);
    public static int round(float);
    public static long round(double);
    public static double random();
    public static int abs(int);
    public static long abs(long);
    public static float abs(float);
    public static double abs(double);
    public static int max(int,int);
    public static long max(long,long);
    public static float max(float,float);
    public static double max(double,double);
    public static int min(int,int);
    public static long min(long,long);
    public static float min(float,float);
    public static double min(double,double);
    public static double ulp(double);
    public static float ulp(float);
    public static double signum(double);
    public static float signum(float);
    public static double sinh(double);
    public static double cosh(double);
    public static double tanh(double);
    public static double hypot(double,double);
    public static double expm1(double);
    public static double log1p(double);
}

# Remove - Number method calls. Remove all invocations of Number
# methods without side effects whose return values are not used.
-assumenosideeffects public class java.lang.* extends java.lang.Number {
    public static java.lang.String toString(byte);
    public static java.lang.Byte valueOf(byte);
    public static byte parseByte(java.lang.String);
    public static byte parseByte(java.lang.String,int);
    public static java.lang.Byte valueOf(java.lang.String,int);
    public static java.lang.Byte valueOf(java.lang.String);
    public static java.lang.Byte decode(java.lang.String);
    public int compareTo(java.lang.Byte);
    public static java.lang.String toString(short);
    public static short parseShort(java.lang.String);
    public static short parseShort(java.lang.String,int);
    public static java.lang.Short valueOf(java.lang.String,int);
    public static java.lang.Short valueOf(java.lang.String);
    public static java.lang.Short valueOf(short);
    public static java.lang.Short decode(java.lang.String);
    public static short reverseBytes(short);
    public int compareTo(java.lang.Short);
    public static java.lang.String toString(int,int);
    public static java.lang.String toHexString(int);
    public static java.lang.String toOctalString(int);
    public static java.lang.String toBinaryString(int);
    public static java.lang.String toString(int);
    public static int parseInt(java.lang.String,int);
    public static int parseInt(java.lang.String);
    public static java.lang.Integer valueOf(java.lang.String,int);
    public static java.lang.Integer valueOf(java.lang.String);
    public static java.lang.Integer valueOf(int);
    public static java.lang.Integer getInteger(java.lang.String);
    public static java.lang.Integer getInteger(java.lang.String,int);
    public static java.lang.Integer getInteger(java.lang.String,java.lang.Integer);
    public static java.lang.Integer decode(java.lang.String);
    public static int highestOneBit(int);
    public static int lowestOneBit(int);
    public static int numberOfLeadingZeros(int);
    public static int numberOfTrailingZeros(int);
    public static int bitCount(int);
    public static int rotateLeft(int,int);
    public static int rotateRight(int,int);
    public static int reverse(int);
    public static int signum(int);
    public static int reverseBytes(int);
    public int compareTo(java.lang.Integer);
    public static java.lang.String toString(long,int);
    public static java.lang.String toHexString(long);
    public static java.lang.String toOctalString(long);
    public static java.lang.String toBinaryString(long);
    public static java.lang.String toString(long);
    public static long parseLong(java.lang.String,int);
    public static long parseLong(java.lang.String);
    public static java.lang.Long valueOf(java.lang.String,int);
    public static java.lang.Long valueOf(java.lang.String);
    public static java.lang.Long valueOf(long);
    public static java.lang.Long decode(java.lang.String);
    public static java.lang.Long getLong(java.lang.String);
    public static java.lang.Long getLong(java.lang.String,long);
    public static java.lang.Long getLong(java.lang.String,java.lang.Long);
    public static long highestOneBit(long);
    public static long lowestOneBit(long);
    public static int numberOfLeadingZeros(long);
    public static int numberOfTrailingZeros(long);
    public static int bitCount(long);
    public static long rotateLeft(long,int);
    public static long rotateRight(long,int);
    public static long reverse(long);
    public static int signum(long);
    public static long reverseBytes(long);
    public int compareTo(java.lang.Long);
    public static java.lang.String toString(float);
    public static java.lang.String toHexString(float);
    public static java.lang.Float valueOf(java.lang.String);
    public static java.lang.Float valueOf(float);
    public static float parseFloat(java.lang.String);
    public static boolean isNaN(float);
    public static boolean isInfinite(float);
    public static int floatToIntBits(float);
    public static int floatToRawIntBits(float);
    public static float intBitsToFloat(int);
    public static int compare(float,float);
    public boolean isNaN();
    public boolean isInfinite();
    public int compareTo(java.lang.Float);
    public static java.lang.String toString(double);
    public static java.lang.String toHexString(double);
    public static java.lang.Double valueOf(java.lang.String);
    public static java.lang.Double valueOf(double);
    public static double parseDouble(java.lang.String);
    public static boolean isNaN(double);
    public static boolean isInfinite(double);
    public static long doubleToLongBits(double);
    public static long doubleToRawLongBits(double);
    public static double longBitsToDouble(long);
    public static int compare(double,double);
    public boolean isNaN();
    public boolean isInfinite();
    public int compareTo(java.lang.Double);
    public byte byteValue();
    public short shortValue();
    public int intValue();
    public long longValue();
    public float floatValue();
    public double doubleValue();
    public int compareTo(java.lang.Object);
    public boolean equals(java.lang.Object);
    public int hashCode();
    public java.lang.String toString();
}

# Remove - String method calls. Remove all invocations of String
# methods without side effects whose return values are not used.
-assumenosideeffects public class java.lang.String {
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

# Remove - StringBuffer method calls. Remove all invocations of StringBuffer
# methods without side effects whose return values are not used.
-assumenosideeffects public class java.lang.StringBuffer {
    public java.lang.String toString();
    public char charAt(int);
    public int capacity();
    public int codePointAt(int);
    public int codePointBefore(int);
    public int indexOf(java.lang.String,int);
    public int lastIndexOf(java.lang.String);
    public int lastIndexOf(java.lang.String,int);
    public int length();
    public java.lang.String substring(int);
    public java.lang.String substring(int,int);
}

# Remove - StringBuilder method calls. Remove all invocations of StringBuilder
# methods without side effects whose return values are not used.
-assumenosideeffects public class java.lang.StringBuilder {
    public java.lang.String toString();
    public char charAt(int);
    public int capacity();
    public int codePointAt(int);
    public int codePointBefore(int);
    public int indexOf(java.lang.String,int);
    public int lastIndexOf(java.lang.String);
    public int lastIndexOf(java.lang.String,int);
    public int length();
    public java.lang.String substring(int);
    public java.lang.String substring(int,int);
}
