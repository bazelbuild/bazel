Zstd-jni
========

[![Build Status](https://api.travis-ci.org/luben/zstd-jni.svg)](https://travis-ci.org/luben/zstd-jni)
[![Windows Build](https://ci.appveyor.com/api/projects/status/qs6j4n9t3bsi5p4w/branch/master?svg=true)](https://ci.appveyor.com/project/luben/zstd-jni)
[![codecov.io](http://codecov.io/github/luben/zstd-jni/coverage.svg?branch=master)](http://codecov.io/github/luben/zstd-jni?branch=master)
[![Code Quality: C](https://img.shields.io/lgtm/grade/cpp/g/luben/zstd-jni.svg?logo=lgtm&logoWidth=18&label=C)](https://lgtm.com/projects/g/luben/zstd-jni/context:cpp)
[![Code Quality: Java](https://img.shields.io/lgtm/grade/java/g/luben/zstd-jni.svg?logo=lgtm&logoWidth=18&label=Java)](https://lgtm.com/projects/g/luben/zstd-jni/context:java)
[![Total Alerts](https://img.shields.io/lgtm/alerts/g/luben/zstd-jni.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/luben/zstd-jni/alerts)
[![Maven Central](https://img.shields.io/maven-central/v/com.github.luben/zstd-jni.svg?label=Maven%20Central)](https://search.maven.org/artifact/com.github.luben/zstd-jni/)
[![Javadocs](https://www.javadoc.io/badge/com.github.luben/zstd-jni.svg)](https://www.javadoc.io/doc/com.github.luben/zstd-jni)

Overview
--------

JNI bindings for **Zstd** native library that provides fast and high
compression lossless algorithm for Android, Java and all JVM languages:

* static compress/decompress methods

* implementation of InputStream and OutputStream for transparent compression
of data streams fully compatible with the "zstd" program.

* minimal performance overhead

Zstd
----

**Zstd**, short for Zstandard, is a new lossless compression algorithm, which
provides both good compression ratio _and_ speed for your standard compression
needs. "Standard" translates into everyday situations which neither look for
highest possible ratio (which LZMA and ZPAQ cover) nor extreme speeds (which
LZ4 covers).

**Zstd** is developed by Yann Collet and the source is available at:
https://github.com/facebook/zstd

The motivation for development, the algorithm used and its properties are
explained in the blog post that introduces the library:
http://fastcompression.blogspot.com/2015/01/zstd-stronger-compression-algorithm.html

Status and availability
-----------------------

**Zstd** is production ready with a stable format.

**Zstd-jni** is tracking the release branch of **Zstd** (master) with
compatibility support for the legacy formats (since version 0.4).

**Zstd-jni** version uses the base **Zstd** version with **Zstd-jni** release
appended with a dash, e.g. "1.2.0-2" is the second **Zstd-jni** release based
on **Zstd** version 1.2.0.

Building and dependencies
-------------------------

**Zstd-jni** uses SBT for building the libary and running the tests.

The build system depends on Scala and the tests depend on ScalaTest and
ScalaCheck but the produced JAR does not have any dependencies. It also
embeds the native library.

How to build:

```
 $ sbt compile test package
```

If you want to publish it to you local ivy2 repository:

```
 $ sbt publish-local
```

Binary releases
---------------

The binary releases are architecture dependent because we are embedding the
native library in the provided Jar file. Currently they are built for
*linux-amd64*, *linux-i386*, *linux-aarch64*, *linux-ppc64*, *linux-ppc64le*,
*linux-mips64*, *win-amd64*, *win-x86*, *aix-ppc64*, *darwin-x86_64* (MacOS X)
and *freebsd-amd64*. More builds will be available if I get access to more
platforms.

You can find published releases on Maven Central.

    <dependency>
        <groupId>com.github.luben</groupId>
        <artifactId>zstd-jni</artifactId>
        <version>VERSION</version>
    </dependency>

sbt dependency:

    libraryDependencies += "com.github.luben" % "zstd-jni" % "VERSION"

Single architecture classified jars are also published. They can be used like:

    <dependency>
        <groupId>com.github.luben</groupId>
        <artifactId>zstd-jni</artifactId>
        <version>VERSION</version>
        <classifier>linux_amd64</classifier>
    </dependency>

or for sbt:

    libraryDependencies += "com.github.luben" % "zstd-jni" % "VERSION" classifier "linux_amd64"

Link for direct download if you don't use a dependency manager:

 - https://repo1.maven.org/maven2/com/github/luben/zstd-jni/

If there is not yet a binary release compatible with your platform look how
to build it locally under the [Building](#building-and-dependencies) section.

Android support
---------------

Zstd-jni is usable in Android applications by importing the sources in Android
Studio. I guess using git sub-modules will also work.

Android archive (*zstd-jni.aar*) is also published on maven central. You will need
to add the repository in your build.gradle, e.g.:

    allprojects {
        repositories {
            jcenter()
            mavenCentral()
        }
    }

as it is not added by default by Android Studio. And then add dependency on the
prebuilt android archive (aar):

    dependencies {
        implementation "com.github.luben:zstd-jni:VERSION@aar"
    }

For example Android app and how to declare dependencies and use zstd-jni, consult
the 2nd and 3rd commit of: https://github.com/luben/ZstdAndroidExample

License
-------

The code for these JNI bindings is licenced under 2-clause BSD license.
The native **Zstd** library is licensed under 3-clause BSD license and
GPL2. See the LICENSE file and LICENSE and COPYRIGHT in src/main/native
for full copyright and conditions.
