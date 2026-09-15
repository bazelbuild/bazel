You can enable ProGuard in your Android Gradle build process, by enabling
minification in the `build.gradle` file of your project:

    android {
        .....
        buildTypes {
            debug {
                minifyEnabled   false
                shrinkResources false
            }
            release {
                minifyEnabled   true
                shrinkResources true
                proguardFile getDefaultProguardFile('proguard-android-optimize.pro')
                proguardFile 'proguard-project.txt'
            }
        }
    }

You then need to make sure that you use ProGuard instead of R8, in the
`gradle.properties` file of your project:

    android.enableR8=false
    android.enableR8.libraries=false

You can find a complete sample project in `examples/android` in the ProGuard
distribution.

## ProGuard Gradle Plugin

Alternatively, you can enable ProGuard in the Android Gradle build process
using ProGuard's own tuned plugin, by changing the `build.gradle` file of your
project as follows:

    buildscript {
        repositories {
            google()
            jcenter()
        }
        dependencies {
            classpath 'com.android.tools.build:gradle:3.3.0'
            classpath 'net.sf.proguard:proguard-gradle:6.2.2'
        }
    }

    apply plugin: 'com.android.application'
    apply plugin: 'proguard'

Or if you want to use your local copy of the plugin:

    buildscript {
        repositories {
            flatDir dirs: '/usr/local/java/proguard/lib'
            google()
            jcenter()
        }
        dependencies {
            classpath 'com.android.tools.build:gradle:3.3.0'
            classpath ':proguard:'
        }
    }

    apply plugin: 'com.android.application'
    apply plugin: 'proguard'

Please make sure the repository path in the build script is set correctly for
your system.

Each build type that should be processed by ProGuard needs to have a set of
configuration files, as shown below:

    android {
        .....
        buildTypes {
            debug {
                minifyEnabled false
                proguardFile getTunedProGuardFile('proguard-android-debug.pro')
                proguardFile 'proguard-project.txt'
            }
            release {
                minifyEnabled false
                proguardFile getTunedProGuardFile('proguard-android-release.pro')
                proguardFile 'proguard-project.txt'
            }
        }
    }

The setting "`minifyEnabled=false`" is needed to disable the
obfuscation/shrinking capability of the standard gradle plugin to avoid that
the project is obfuscated multiple times.

The lines with "`proguardFile getTunedProGuardFile`" are important. They apply
optimized minimal settings for the Android platform. Your own configuration
files then only need to contain project-specific settings.

You can find a complete sample project in `examples/android-plugin` in the
ProGuard distribution.

## Settings {: #proguard}

The **ProGuard plugin** supports various settings that can be added to the
*build.gradle* file to control its behavior:

    proguard {
        incremental                false
        transformExternalLibraries true
        transformSubprojects       true
    }

`incremental:`
: Support incremental builds, default: `false`

`transformExternalLibraries:`
: Processing also all external libraries, default: `true`

`transformSubproject:`
: Processing also all subprojects, default: `true`

## Java 8 language support

ProGuard can backport Java 8 language features to Dalvik and to older versions
of Java, so you can use them in applications and libraries that work on all
versions of Android.

The following language features are supported:

- lambda expressions (closures)
- method references
- try-with-resources statements
- static interface methods
- default interface methods

Additions to the Java runtime library (like the stream API) can not be
backported without additional runtime dependencies; see the next section.

In order to enable Java 8 language support, a Java 8 Compiler must be
available when running gradle, i.e. by setting `JAVA\_HOME`. You need to
configure the *source* and *target* compatibility settings in your
`build.gradle` script:

    android {
        .....
        compileOptions {
            sourceCompatibility JavaVersion.VERSION_1_8
            targetCompatibility JavaVersion.VERSION_1_8
        }
    }

If you want to target a minimum API level less than 19, you should add the
following configuration to your `proguard-project.txt`:

    -target 1.6

## Java 8 stream API support

ProGuard can backport the Java 8 stream API to older versions of Android (<
API 24), so you can use it in applications that work on all versions of
Android.

The following additions to Java 8 can be backported and thus be used in an
application:

- classes added in `java.util.function` package
- classes added in `java.util.stream` package
- default and static methods added to interfaces in `java.lang` and
  `java.util` (e.g. Iterable, Collection)
  - static methods added to classes in `java.lang` (Integer, Long, Double) and
  `java.util` (Arrays)
- classes added in `java.util` package (e.g. Optional, StringJoiner)
- classes added/updated in `java.util.concurrent` package (e.g. ForkJoinPool)

The following methods can not be backported and using them will fail the
build:

- methods added to `java.util.Random` (e.g. ints, longs, doubles)
- default method `remove()` in `java.util.Iterator`: classes that rely on a
  default implementation will not work

In order to use the Java 8 stream API and enable backporting in ProGuard, you
have to specify a minimum `compileSdkVersion` of *24*:

    android {
        .....
        compileSdkVersion 24 // or a higher version
    }

Furthermore, you need to add the *streamsupport* library as dependency to your
project, e.g. like this:

    repositories {
        jcenter()
    }

    dependencies {
        ...
        compile 'net.sourceforge.streamsupport:streamsupport:1.6.3'
    }

If you use the Java 8 stream API in your project, but ProGuard can not
backport to the included *streamsupport* library (e.g. due to an incompatible
version), ProGuard issues warnings in the console log.

Note: ProGuard currently only supports the core streamsupport library.
Additional modules (cfuture, atomic, flow, literal) are not yet backported.
Please contact [Guardsquare](https://www.guardsquare.com/) if you want to use
them as well.

## Java 8 time API support (JSR310)

ProGuard can backport the **Java 8 time API** to older versions of Android
(< API 26), so you can use it in applications that work on all versions of
Android.

The following additions to Java 8 can be backported and thus be used in an
application:

- classes added in `java.time` package

The *java.time* API includes various interfaces with default / static methods.
ProGuard can backport the use of these methods can be backported, but in case
you implement these interfaces yourself (which would be unusual), the
resulting class can not be fully backported (due to missing default methods).
This includes the following interfaces:

- java.time.chrono.ChronoLocalDate
- java.time.chrono.ChronoLocalDateTime
- java.time.chrono.Chronology
- java.time.chrono.ChronoPeriod
- java.time.chrono.ChronoZonedDateTime
- java.time.chrono.Era
- java.time.temporal.Temporal
- java.time.temporal.TemporalAccessor
- java.time.temporal.TemporalField
- java.time.temporal.TemporalUnit

In order to use the Java 8 time API and enable backporting in ProGuard,
a minimum `compileSdkVersion` of *26* has to be used:

    android {
        .....
        compileSdkVersion 26 // or a higher version
    }

Furthermore, you need to add the *threetenbp* library as dependency to your
project, e.g. like this:

    repositories {
        jcenter()
    }

    dependencies {
        ...
        api group: 'org.threeten', name: 'threetenbp', version: '1.3.6'
    }

It is highly recommended to use the *threetenabp* library from Jake Wharton
instead, which is just as a wrapper around the *threetenbp* library and
provides an efficient way to initialize the timezone database:

    dependencies {
        ...
        compile 'com.jakewharton.threetenabp:threetenabp:1.0.5'
    }

Additionally, you have to initialize the timezone database in your application
class like this:

    public void onCreate()
    {
        super.onCreate();
        ...
        com.jakewharton.threetenabp.AndroidThreeTen.init(this);
    }

If you use the Java 8 stream API in your project, but ProGuard can not
backport to the included *threetenbp* library (e.g. due to an incompatible
version), ProGuard issues warnings in the console log.
