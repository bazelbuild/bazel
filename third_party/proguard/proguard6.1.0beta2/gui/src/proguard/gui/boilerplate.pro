# Keep - Applications. Keep all application classes, along with their 'main' methods.
-keepclasseswithmembers public class * {
    public static void main(java.lang.String[]);
}

# Keep - Applets. Keep all extensions of java.applet.Applet.
-keep public class * extends java.applet.Applet

# Keep - Servlets. Keep all extensions of javax.servlet.Servlet.
-keep public class * extends javax.servlet.Servlet

# Keep - Midlets. Keep all extensions of javax.microedition.midlet.MIDlet.
-keep public class * extends javax.microedition.midlet.MIDlet

# Keep - Xlets. Keep all extensions of javax.tv.xlet.Xlet.
-keep public class * extends javax.tv.xlet.Xlet

# Keep - Libraries. Keep all public and protected classes, fields, and methods.
-keep public class * {
    public protected <fields>;
    public protected <methods>;
}

# Keep - Native method names. Keep all native class/method names.
-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}

# Keep - _class method names. Keep all .class method names. This may be
# useful for libraries that will be obfuscated again with different obfuscators.
-keepclassmembernames class * {
    java.lang.Class class$(java.lang.String);
    java.lang.Class class$(java.lang.String,boolean);
}

# Also keep - Enumerations. Keep the special static methods that are required in
# enumeration classes.
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Also keep - Serialization code. Keep all fields and methods that are used for
# serialization.
-keepclassmembers class * extends java.io.Serializable {
    static final long serialVersionUID;
    static final java.io.ObjectStreamField[] serialPersistentFields;
    private void writeObject(java.io.ObjectOutputStream);
    private void readObject(java.io.ObjectInputStream);
    java.lang.Object writeReplace();
    java.lang.Object readResolve();
}

# Also keep - BeanInfo classes. Keep all implementations of java.beans.BeanInfo.
-keep class * implements java.beans.BeanInfo

# Also keep - Bean classes. Keep all specified classes, along with their getters
# and setters.
-keep class * {
    void set*(***);
    void set*(int,***);

    boolean is*();
    boolean is*(int);

    *** get*();
    *** get*(int);
}

# Also keep - Database drivers. Keep all implementations of java.sql.Driver.
-keep class * implements java.sql.Driver

# Also keep - Swing UI L&F. Keep all extensions of javax.swing.plaf.ComponentUI,
# along with the special 'createUI' method.
-keep class * extends javax.swing.plaf.ComponentUI {
    public static javax.swing.plaf.ComponentUI createUI(javax.swing.JComponent);
}

# Also keep - RMI interfaces. Keep all interfaces that extend the
# java.rmi.Remote interface, and their methods.
-keep interface * extends java.rmi.Remote {
    <methods>;
}

# Also keep - RMI implementations. Keep all implementations of java.rmi.Remote,
# including any explicit or implicit implementations of Activatable, with their
# two-argument constructors.
-keep class * implements java.rmi.Remote {
    <init>(java.rmi.activation.ActivationID,java.rmi.MarshalledObject);
}



# Android - Android activities. Keep all extensions of Android activities.
-keep public class * extends android.app.Activity

# Android - Android applications. Keep all extensions of Android applications.
-keep public class * extends android.app.Application

# Android - Android services. Keep all extensions of Android services.
-keep public class * extends android.app.Service

# Android - Broadcast receivers. Keep all extensions of Android broadcast receivers.
-keep public class * extends android.content.BroadcastReceiver

# Android - Content providers. Keep all extensions of Android content providers.
-keep public class * extends android.content.ContentProvider


# More Android - View classes. Keep all Android views and their constructors and setters.
-keep public class * extends android.view.View {
      public <init>(android.content.Context);
      public <init>(android.content.Context, android.util.AttributeSet);
      public <init>(android.content.Context, android.util.AttributeSet, int);
      public void set*(...);
}

# More Android - Layout classes. Keep classes with constructors that may be referenced from Android layout
# files.
-keepclasseswithmembers class * {
    public <init>(android.content.Context, android.util.AttributeSet);
 }
-keepclasseswithmembers class * {
     public <init>(android.content.Context, android.util.AttributeSet, int);
}

# More Android - Contexts. Keep all extensions of Android Context.
-keepclassmembers class * extends android.content.Context {
    public void *(android.view.View);
    public void *(android.view.MenuItem);
}

# More Android - Parcelables. Keep all extensions of Android Parcelables.
-keepclassmembers class * implements android.os.Parcelable {
    static ** CREATOR;
}

# More Android - R classes. Keep all fields of Android R classes.
-keepclassmembers class **.R$* {
    public static <fields>;
}


# Android annotations - Support annotations. Support annotations for Android.
-keep @android.support.annotation.Keep class *
-keepclassmembers class * {
    @android.support.annotation.Keep *;
}

# Android annotations - Facebook keep annotations. Keep annotations for Facebook.
-keep @com.facebook.proguard.annotations.DoNotStrip class *
-keepclassmembers class * {
    @com.facebook.proguard.annotations.DoNotStrip *;
}
-keep @com.facebook.proguard.annotations.KeepGettersAndSetters class *
-keepclassmembers class * {
    @com.facebook.proguard.annotations.KeepGettersAndSetters *;
}

# Android annotations - ProGuard annotations. Keep annotations for ProGuard.
-keep @proguard.annotation.Keep class *
-keepclassmembers class * {
    @proguard.annotation.Keep *;
}
-keepclassmembernames class * {
    @proguard.annotation.KeepName *;
}
-keep class * implements @proguard.annotation.KeepImplementations *
-keep public class * implements @proguard.annotation.KeepPublicImplementations *
-keepclassmembers @proguard.annotation.KeepClassMembers class * {
    *;
}
-keepclassmembers @proguard.annotation.KeepPublicClassMembers class * {
    public *;
}
-keepclassmembers @proguard.annotation.KeepPublicProtectedClassMembers class * {
    public protected *;
}
-keepclassmembernames @proguard.annotation.KeepClassMemberNames class * {
    *;
}
-keepclassmembernames @proguard.annotation.KeepPublicClassMemberNames class * {
    public *;
}
-keepclassmembernames @proguard.annotation.KeepPublicProtectedClassMemberNames class * {
    public protected *;
}
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
-keepnames @proguard.annotation.KeepName class *

# Android annotations - Google keep annotations. Keep annotations for Google.
-keepclassmembernames class * {
    @com.google.android.gms.common.annotation.KeepName *;
}
-keepnames @com.google.android.gms.common.annotation.KeepName class *



# Android libraries - Design support libraries. Keep setters for design support libraries.
-keep !abstract class android.support.design.widget.* implements
android.support.design.widget.CoordinatorLayout$Behavior {
    <init>(android.content.Context, android.util.AttributeSet);
}
-keepnames class android.support.design.widget.CoordinatorLayout

# Android libraries - Kotlin. Keep some methods for Kotlin for Android Development.
-keepclassmembers,allowshrinking,allowobfuscation class
kotlin.jvm.internal.Intrinsics {
    void throwNpe();
}

# Android libraries - Google Play Services. Keep classes for Google Play Services.
-keep class com.google.android.gms.tagmanager.TagManagerService
-keep class com.google.android.gms.measurement.AppMeasurement
-keep class com.google.android.gms.measurement.AppMeasurementReceiver
-keep class com.google.android.gms.measurement.AppMeasurementService
-keepclassmembers class
    com.google.android.gms.ads.identifier.AdvertisingIdClient,
    com.google.android.gms.ads.identifier.AdvertisingIdClient$Info,
    com.google.android.gms.common.GooglePlayServicesUtil {
    public <methods>;
}
-keep,allowobfuscation class
    com.google.android.gms.ads.identifier.AdvertisingIdClient,
    com.google.android.gms.ads.identifier.AdvertisingIdClient$Info,
    com.google.android.gms.common.GooglePlayServicesUtil
-keep,allowshrinking class
    com.google.android.gms.iid.MessengerCompat,
    com.google.android.gms.location.ActivityRecognitionResult,
    com.google.android.gms.maps.GoogleMapOptions
-keep,allowshrinking class
    com.google.android.gms.ads.AdActivity,
    com.google.android.gms.ads.purchase.InAppPurchaseActivity,
    com.google.android.gms.gcm.GoogleCloudMessaging,
    com.google.android.gms.location.places.*Api
-keepclassmembers class
com.google.android.gms.common.internal.safeparcel.SafeParcelable {
    public static final java.lang.String NULL;
}

# Android libraries - Firebase. Keep classes for Firebase for Android.
-keep class com.google.firebase.FirebaseApp
-keep class com.google.firebase.auth.FirebaseAuth
-keep class com.google.firebase.crash.FirebaseCrash
-keep class
com.google.firebase.database.connection.idl.IPersistentConnectionImpl
-keep class com.google.firebase.iid.FirebaseInstanceId

# Android libraries - Google Cloud Messaging. Keep classes for Google Cloud Messaging.
-keep,allowshrinking class **.GCMIntentService

# Android libraries - Guava. Keep classes for the Guava libraries.
-keepclassmembers class com.google.common.primitives.UnsignedBytes$LexicographicalComparatorHolder$UnsafeComparator
{
    sun.misc.Unsafe theUnsafe;
}

# Android libraries - RxJava. Keep classes for RxJava.
-keepclassmembers class rx.internal.util.unsafe.*Queue {
    long producerIndex;
    long consumerIndex;

    rx.internal.util.atomic.LinkedQueueNode producerNode;
    rx.internal.util.atomic.LinkedQueueNode consumerNode;
}

# Android libraries - ActionBarSherlock. Keep classes for ActionBarSherlock.
-keepclassmembers !abstract class * extends
com.actionbarsherlock.ActionBarSherlock {
    <init>(android.app.Activity, int);
}

# Android libraries - GSON. Keep classes for the GSON library.
-keepclassmembers class * {
    @com.google.gson.annotations.Expose <fields>;
}
-keepclassmembers enum * {
    @com.google.gson.annotations.SerializedName <fields>;
}
-keepclasseswithmembers,allowobfuscation,includedescriptorclasses class * {
    @com.google.gson.annotations.Expose <fields>;
}
-keepclasseswithmembers,allowobfuscation,includedescriptorclasses class * {
    @com.google.gson.annotations.SerializedName <fields>;
}


# Android libraries - Dagger code. Keep the classes that Dagger accesses
# by reflection.
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

-keepclassmembers,allowobfuscation class * {
    @dagger.** *;
}

# Android libraries - Butterknife code. Keep the classes that Butterknife accesses
# by reflection.
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

-keep,allowobfuscation @interface butterknife.*
-dontwarn butterknife.internal.ButterKnifeProcessor

# Android libraries - Roboguice. Keep classes for RoboGuice.
-keepclassmembers class * implements com.google.inject.Module {
    <init>(android.content.Context);
    <init>();
}
-keepclassmembers class android.support.v4.app.Fragment {
    public android.view.View getView();
}
-keepclassmembers class android.support.v4.app.FragmentManager {
    public android.support.v4.app.Fragment findFragmentById(int);
    public android.support.v4.app.Fragment
findFragmentByTag(java.lang.String);
}
-keep,allowobfuscation class roboguice.activity.event.OnCreateEvent
-keep,allowobfuscation class roboguice.inject.SharedPreferencesProvider$PreferencesNameHolder
-dontnote com.google.inject.Module

# Android libraries - Otto. Keep classes for Otto.
-keepclassmembers,allowobfuscation class * {
    @com.squareup.otto.* <methods>;
}

# Android libraries - Greenrobot EventBus V2.
-keepclassmembers class * {
    public void onEvent*(***);
}

# Android libraries - Greenrobot EventBus V3.
-keep enum org.greenrobot.eventbus.ThreadMode { *; }
-keepclassmembers class * extends
org.greenrobot.eventbus.util.ThrowableFailureEvent {
    <init>(java.lang.Throwable);
}
-keep,allowobfuscation class org.greenrobot.eventbus.Subscribe { *; }
-keepclassmembers,allowobfuscation class ** {
    @org.greenrobot.eventbus.Subscribe <methods>;
}
# Android libraries - Google API. Keep classes and field for Google API.
-keepclassmembers class * {
    @com.google.api.client.util.Key       <fields>;
    @com.google.api.client.util.Value     <fields>;
    @com.google.api.client.util.NullValue <fields>;
}
-keep,allowobfuscation class com.google.api.client.util.Types {
    java.lang.IllegalArgumentException
handleExceptionForNewInstance(java.lang.Exception, java.lang.Class);
}

# Android libraries - Facebook API. Keep methods for the Facebook API.
-keepclassmembers interface com.facebook.model.GraphObject {
    <methods>;
}

# Android libraries - Javascript interfaces. Keep all methods from Android Javascripts.
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}

# Android libraries - Compatibility classes. Avoid merging and inlining compatibility classes.
-keep,allowshrinking,allowobfuscation class
    android.support.**Compat*,
    android.support.**Honeycomb*,
    android.support.**IceCreamSandwich*,
    android.support.**JellyBean*,
    android.support.**Jellybean*,
    android.support.**JB*,
    android.support.**KitKat*,
    android.support.**Kitkat*,
    android.support.**Lollipop*,
    android.support.**19,
    android.support.**20,
    android.support.**21,
    android.support.**22,
    android.support.**23,
    android.support.**24,
    android.support.**25,
    android.support.**Api*
        { *; }
-keep,allowobfuscation,allowshrinking @android.annotation.TargetApi
class android.support.** { *; }

# Android libraries - Signatures. Keep setters for signature optimized with class from API level 19 or
# higher.
-keep,allowshrinking,allowobfuscation class
android.support.v4.app.FragmentState$InstantiationException {
    <init>(...);
}
-keep,allowshrinking,allowobfuscation class
android.support.v4.app.Fragment$InstantiationException {
    <init>(...);
}

# Android libraries - Fields accessed before initialization. Keep fields that are accessed before
# initialization.
-keepclassmembers,allowshrinking,allowobfuscation class
android.support.**,com.google.android.gms.**.internal.** {
    !static final <fields>;
}

# Android libraries - Sensitive classes. Keep sensitive classes.
-keep,allowshrinking,allowobfuscation class
com.google.android.gms.**.z* { *; }

# Android libraries - Injection. Keep classes for injection in Guice/RoboGuice/Dagger/ActionBarSherlock.
-keep,allowobfuscation class * implements com.google.inject.Provider
-keep,allowobfuscation @interface javax.inject.**          { *; }
-keep,allowobfuscation @interface com.google.inject.**     { *; }
-keep,allowobfuscation @interface roboguice.**             { *; }
-keep,allowobfuscation @interface com.actionbarsherlock.** { *; }
-keepclassmembers,allowobfuscation class * {
    @javax.inject.**          <fields>;
    @com.google.inject.**     <fields>;
    @roboguice.**             <fields>;
    @roboguice.event.Observes <methods>;
    @com.actionbarsherlock.** <fields>;
    @dagger.**                *;
    !private <init>();
    @com.google.inject.Inject <init>(...);
}

# Android libraries - Retrofit. Keep classes for Retrofit.
-keepclassmembers @retrofit.http.RestMethod @interface * {
    <methods>;
}
-keepclassmembers,allowobfuscation interface * {
    @retrofit.http.** <methods>;
}
-keep,allowobfuscation @retrofit.http.RestMethod @interface *
-keep,allowobfuscation @interface retrofit2.http.**

# Android libraries - Google inject. Keep classes for Google inject.
-keepclassmembers class * {
    void finalizeReferent();
}
-keepclassmembers class com.google.inject.internal.util.$Finalizer {
    public static java.lang.ref.ReferenceQueue
startFinalizer(java.lang.Class,java.lang.Object);
}
-keepnames class com.google.inject.internal.util.$FinalizableReference


# Android libraries - Jackson. Keep classes for Jackson.
-keepclassmembers class * {
    @org.codehaus.jackson.annotate.* <methods>;
}
-keepclassmembers @org.codehaus.jackson.annotate.JsonAutoDetect class * {
    void set*(***);
    *** get*();
    boolean is*();
}

# Android libraries - Crashlytics. Keep classes for Crashlytics.
-keep class * extends io.fabric.sdk.android.Kit



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

# Remove debugging - Throwable_printStackTrace calls. Remove all invocations of
# Throwable.printStackTrace().
-assumenosideeffects public class java.lang.Throwable {
    public void printStackTrace();
}

# Remove debugging - Thread_dumpStack calls. Remove all invocations of
# Thread.dumpStack().
-assumenosideeffects public class java.lang.Thread {
    public static void dumpStack();
}

# Remove debugging - All logging API calls. Remove all invocations of the
# logging API whose return values are not used.
-assumenosideeffects public class java.util.logging.* {
    <methods>;
}

# Remove debugging - All Log4j API calls. Remove all invocations of the
# Log4j API whose return values are not used.
-assumenosideeffects public class org.apache.log4j.** {
    <methods>;
}

