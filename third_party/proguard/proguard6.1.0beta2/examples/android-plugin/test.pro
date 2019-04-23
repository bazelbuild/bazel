-injars /home/eric/ProGuard/proguard/examples/android-plugin/build/intermediates/javac/release/compileReleaseJavaWithJavac/classes(**.class)
-outjars /home/eric/ProGuard/proguard/examples/android-plugin/build/intermediates/transforms/ProGuardTransform/release/0

-libraryjars /usr/local/java/android-sdk/platforms/android-28/android.jar
-libraryjars /usr/local/java/android-sdk/platforms/android-28/optional/org.apache.http.legacy.jar
-libraryjars /usr/local/java/android-sdk/platforms/android-28/optional/android.test.mock.jar
-libraryjars /usr/local/java/android-sdk/platforms/android-28/optional/android.test.base.jar
-libraryjars /usr/local/java/android-sdk/platforms/android-28/optional/android.test.runner.jar

-target 1.6
-forceprocessing
-printusage /home/eric/ProGuard/proguard/examples/android-plugin/build/outputs/mapping/release/usage.txt
-dontoptimize
-printmapping /home/eric/ProGuard/proguard/examples/android-plugin/build/outputs/mapping/release/mapping.txt
-dontusemixedcaseclassnames
-keepattributes *Annotation*
-dontpreverify
-android
-verbose
-dontnote android.support.v4.app.Fragment,com.google.vending.licensing.ILicensingService,com.android.vending.licensing.ILicensingService,android.support.annotation.Keep
-dontwarn sun.**,javax.**,java.awt.**,java.nio.file.**,org.apache.**,build.IgnoreJava8API,android.support.**
-ignorewarnings
-printconfiguration /home/eric/ProGuard/proguard/examples/android-plugin/test.pro
-printseeds /home/eric/ProGuard/proguard/examples/android-plugin/build/outputs/mapping/release/seeds.txt



-keep public class * extends android.app.Activity

-keep public class * extends android.app.Application

-keep public class * extends android.app.Service

-keep public class * extends android.content.BroadcastReceiver

-keep public class * extends android.content.ContentProvider

-keep public class * extends android.app.backup.BackupAgent

-keep public class * extends android.preference.Preference

-keep public class * extends android.support.v4.app.Fragment

-keep public class * extends android.app.Fragment

-keep public class com.google.vending.licensing.ILicensingService

-keep public class com.android.vending.licensing.ILicensingService

# From the default AGP config: keep constructors that are called from
# the system via reflection.
-keep public class * extends android.view.View {
    public <init>(android.content.Context);
    public <init>(android.content.Context,android.util.AttributeSet);
    public <init>(android.content.Context,android.util.AttributeSet,int);
    public void set*(...);
}

-keepclasseswithmembers class * {
    public <init>(android.content.Context,android.util.AttributeSet);
}

-keepclasseswithmembers class * {
    public <init>(android.content.Context,android.util.AttributeSet,int);
}

# For native methods, see http://proguard.sourceforge.net/manual/examples.html#native
-keepclasseswithmembers,allowshrinking class * {
    native <methods>;
}

# keep setters in Views so that animations can still work.
# see http://proguard.sourceforge.net/manual/examples.html#beans
-keepclassmembers public class * extends android.view.View {
    void set*(***);
    *** get*();
}

# For enumeration classes, see http://proguard.sourceforge.net/manual/examples.html#enumerations
-keepclassmembers enum  * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

-keepclassmembers class * extends android.os.Parcelable {
    public static final android.os.Parcelable$Creator CREATOR;
}

-keep class android.support.annotation.Keep

-keep @android.support.annotation.Keep class * {
    <fields>;
    <methods>;
}

-keepclasseswithmembers class * {
    @android.support.annotation.Keep
    <methods>;
}

-keepclasseswithmembers class * {
    @android.support.annotation.Keep
    <fields>;
}

-keepclasseswithmembers class * {
    @android.support.annotation.Keep
    <init>(...);
}

# ##############################################################################
# Settings to handle reflection in the code.
# ##############################################################################
# Preserve annotated and generated classes for Dagger.
-keepclassmembers,allowobfuscation class * {
    @dagger.**
    <fields>;
    @dagger.**
    <methods>;
}

-keep class **$$ModuleAdapter

-keep class **$$InjectAdapter

-keep class **$$StaticInjection

-if class **$$ModuleAdapter

-keep class <1>

-if class **$$InjectAdapter

-keep class <1>

-if class **$$StaticInjection

-keep class <1>

-keep,allowshrinking class dagger.Lazy

# Preserve annotated and generated classes for Butterknife.
-keep class **$$ViewBinder {
    public static void bind(...);
    public static void unbind(...);
}

-if class **$$ViewBinder

-keep class <1>

-keep class **_ViewBinding {
    <init>(<1>,android.view.View);
}

-if class **_ViewBinding

-keep class <1>

# Preserve fields that are serialized with GSON.
# -keepclassmembers class com.example.SerializedClass1,
#                        com.example.SerializedClass2 {
#    <fields>;
# }
-keepclassmembers,allowobfuscation class * {
    @com.google.gson.annotations.SerializedName
    <fields>;
}

-keep,allowobfuscation @interface  com.google.gson.annotations.**

# Referenced at /home/eric/ProGuard/proguard/examples/android-plugin/build/intermediates/merged_manifests/release/processReleaseManifest/merged/AndroidManifest.xml:15
-keep class com.example.HelloWorldActivity {
    <init>(...);
}

# ##############################################################################
# Further optimizations.
# ##############################################################################
# If you wish, you can let the optimization step remove Android logging calls.
# -assumenosideeffects class android.util.Log {
#    public static boolean isLoggable(java.lang.String, int);
#    public static int v(...);
#    public static int i(...);
#    public static int w(...);
#    public static int d(...);
#    public static int e(...);
# }
# In that case, it's especially useful to also clean up any corresponding
# string concatenation calls.
-assumenoexternalsideeffects class java.lang.StringBuilder {
    public <init>();
    public <init>(int);
    public <init>(java.lang.String);
    public java.lang.StringBuilder append(java.lang.Object);
    public java.lang.StringBuilder append(java.lang.String);
    public java.lang.StringBuilder append(java.lang.StringBuffer);
    public java.lang.StringBuilder append(char[]);
    public java.lang.StringBuilder append(char[],int,int);
    public java.lang.StringBuilder append(boolean);
    public java.lang.StringBuilder append(char);
    public java.lang.StringBuilder append(int);
    public java.lang.StringBuilder append(long);
    public java.lang.StringBuilder append(float);
    public java.lang.StringBuilder append(double);
    public java.lang.String toString();
}

-assumenoexternalreturnvalues class java.lang.StringBuilder {
    public java.lang.StringBuilder append(java.lang.Object);
    public java.lang.StringBuilder append(java.lang.String);
    public java.lang.StringBuilder append(java.lang.StringBuffer);
    public java.lang.StringBuilder append(char[]);
    public java.lang.StringBuilder append(char[],int,int);
    public java.lang.StringBuilder append(boolean);
    public java.lang.StringBuilder append(char);
    public java.lang.StringBuilder append(int);
    public java.lang.StringBuilder append(long);
    public java.lang.StringBuilder append(float);
    public java.lang.StringBuilder append(double);
}
