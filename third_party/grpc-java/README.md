gRPC-Java - An RPC library and framework
========================================

gRPC-Java works with JDK 8. gRPC-Java clients are supported on Android API
levels 19 and up (KitKat and later). Deploying gRPC servers on an Android
device is not supported.

TLS usage typically requires using Java 8, or Play Services Dynamic Security
Provider on Android. Please see the [Security Readme](SECURITY.md).

<table>
  <tr>
    <td><b>Homepage:</b></td>
    <td><a href="https://grpc.io/">grpc.io</a></td>
  </tr>
  <tr>
    <td><b>Mailing List:</b></td>
    <td><a href="https://groups.google.com/forum/#!forum/grpc-io">grpc-io@googlegroups.com</a></td>
  </tr>
</table>

[![Join the chat at https://gitter.im/grpc/grpc](https://badges.gitter.im/grpc/grpc.svg)](https://gitter.im/grpc/grpc?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/grpc/grpc-java.svg?branch=master)](https://travis-ci.org/grpc/grpc-java)
[![Line Coverage Status](https://coveralls.io/repos/grpc/grpc-java/badge.svg?branch=master&service=github)](https://coveralls.io/github/grpc/grpc-java?branch=master)
[![Branch-adjusted Line Coverage Status](https://codecov.io/gh/grpc/grpc-java/branch/master/graph/badge.svg)](https://codecov.io/gh/grpc/grpc-java)

Getting Started
---------------

For a guided tour, take a look at the [quick start
guide](https://grpc.io/docs/languages/java/quickstart) or the more explanatory [gRPC
basics](https://grpc.io/docs/languages/java/basics).

The [examples](https://github.com/grpc/grpc-java/tree/v1.47.0/examples) and the
[Android example](https://github.com/grpc/grpc-java/tree/v1.47.0/examples/android)
are standalone projects that showcase the usage of gRPC.

Download
--------

Download [the JARs][]. Or for Maven with non-Android, add to your `pom.xml`:
```xml
<dependency>
  <groupId>io.grpc</groupId>
  <artifactId>grpc-netty-shaded</artifactId>
  <version>1.47.0</version>
  <scope>runtime</scope>
</dependency>
<dependency>
  <groupId>io.grpc</groupId>
  <artifactId>grpc-protobuf</artifactId>
  <version>1.47.0</version>
</dependency>
<dependency>
  <groupId>io.grpc</groupId>
  <artifactId>grpc-stub</artifactId>
  <version>1.47.0</version>
</dependency>
<dependency> <!-- necessary for Java 9+ -->
  <groupId>org.apache.tomcat</groupId>
  <artifactId>annotations-api</artifactId>
  <version>6.0.53</version>
  <scope>provided</scope>
</dependency>
```

Or for Gradle with non-Android, add to your dependencies:
```gradle
runtimeOnly 'io.grpc:grpc-netty-shaded:1.47.0'
implementation 'io.grpc:grpc-protobuf:1.47.0'
implementation 'io.grpc:grpc-stub:1.47.0'
compileOnly 'org.apache.tomcat:annotations-api:6.0.53' // necessary for Java 9+
```

For Android client, use `grpc-okhttp` instead of `grpc-netty-shaded` and
`grpc-protobuf-lite` instead of `grpc-protobuf`:
```gradle
implementation 'io.grpc:grpc-okhttp:1.47.0'
implementation 'io.grpc:grpc-protobuf-lite:1.47.0'
implementation 'io.grpc:grpc-stub:1.47.0'
compileOnly 'org.apache.tomcat:annotations-api:6.0.53' // necessary for Java 9+
```

[the JARs]:
https://search.maven.org/search?q=g:io.grpc%20AND%20v:1.47.0

Development snapshots are available in [Sonatypes's snapshot
repository](https://oss.sonatype.org/content/repositories/snapshots/).

Generated Code
--------------

For protobuf-based codegen, you can put your proto files in the `src/main/proto`
and `src/test/proto` directories along with an appropriate plugin.

For protobuf-based codegen integrated with the Maven build system, you can use
[protobuf-maven-plugin][] (Eclipse and NetBeans users should also look at
`os-maven-plugin`'s
[IDE documentation](https://github.com/trustin/os-maven-plugin#issues-with-eclipse-m2e-or-other-ides)):
```xml
<build>
  <extensions>
    <extension>
      <groupId>kr.motd.maven</groupId>
      <artifactId>os-maven-plugin</artifactId>
      <version>1.6.2</version>
    </extension>
  </extensions>
  <plugins>
    <plugin>
      <groupId>org.xolstice.maven.plugins</groupId>
      <artifactId>protobuf-maven-plugin</artifactId>
      <version>0.6.1</version>
      <configuration>
        <protocArtifact>com.google.protobuf:protoc:3.19.2:exe:${os.detected.classifier}</protocArtifact>
        <pluginId>grpc-java</pluginId>
        <pluginArtifact>io.grpc:protoc-gen-grpc-java:1.47.0:exe:${os.detected.classifier}</pluginArtifact>
      </configuration>
      <executions>
        <execution>
          <goals>
            <goal>compile</goal>
            <goal>compile-custom</goal>
          </goals>
        </execution>
      </executions>
    </plugin>
  </plugins>
</build>
```

[protobuf-maven-plugin]: https://www.xolstice.org/protobuf-maven-plugin/

For non-Android protobuf-based codegen integrated with the Gradle build system,
you can use [protobuf-gradle-plugin][]:
```gradle
plugins {
    id 'com.google.protobuf' version '0.8.18'
}

protobuf {
  protoc {
    artifact = "com.google.protobuf:protoc:3.19.2"
  }
  plugins {
    grpc {
      artifact = 'io.grpc:protoc-gen-grpc-java:1.47.0'
    }
  }
  generateProtoTasks {
    all()*.plugins {
      grpc {}
    }
  }
}
```

[protobuf-gradle-plugin]: https://github.com/google/protobuf-gradle-plugin

The prebuilt protoc-gen-grpc-java binary uses glibc on Linux. If you are
compiling on Alpine Linux, you may want to use the [Alpine grpc-java package][]
which uses musl instead.

[Alpine grpc-java package]: https://pkgs.alpinelinux.org/package/edge/testing/x86_64/grpc-java

For Android protobuf-based codegen integrated with the Gradle build system, also
use protobuf-gradle-plugin but specify the 'lite' options:

```gradle
plugins {
    id 'com.google.protobuf' version '0.8.18'
}

protobuf {
  protoc {
    artifact = "com.google.protobuf:protoc:3.19.2"
  }
  plugins {
    grpc {
      artifact = 'io.grpc:protoc-gen-grpc-java:1.47.0'
    }
  }
  generateProtoTasks {
    all().each { task ->
      task.builtins {
        java { option 'lite' }
      }
      task.plugins {
        grpc { option 'lite' }
      }
    }
  }
}

```

API Stability
-------------

APIs annotated with `@Internal` are for internal use by the gRPC library and
should not be used by gRPC users. APIs annotated with `@ExperimentalApi` are
subject to change in future releases, and library code that other projects
may depend on should not use these APIs.

We recommend using the
[grpc-java-api-checker](https://github.com/grpc/grpc-java-api-checker)
(an [Error Prone](https://github.com/google/error-prone) plugin)
to check for usages of `@ExperimentalApi` and `@Internal` in any library code
that depends on gRPC. It may also be used to check for `@Internal` usage or
unintended `@ExperimentalApi` consumption in non-library code.

How to Build
------------

If you are making changes to gRPC-Java, see the [compiling
instructions](COMPILING.md).

High-level Components
---------------------

At a high level there are three distinct layers to the library: *Stub*,
*Channel*, and *Transport*.

### Stub

The Stub layer is what is exposed to most developers and provides type-safe
bindings to whatever datamodel/IDL/interface you are adapting. gRPC comes with
a [plugin](https://github.com/google/grpc-java/blob/master/compiler) to the
protocol-buffers compiler that generates Stub interfaces out of `.proto` files,
but bindings to other datamodel/IDL are easy and encouraged.

### Channel

The Channel layer is an abstraction over Transport handling that is suitable for
interception/decoration and exposes more behavior to the application than the
Stub layer. It is intended to be easy for application frameworks to use this
layer to address cross-cutting concerns such as logging, monitoring, auth, etc.

### Transport

The Transport layer does the heavy lifting of putting and taking bytes off the
wire. The interfaces to it are abstract just enough to allow plugging in of
different implementations. Note the transport layer API is considered internal
to gRPC and has weaker API guarantees than the core API under package `io.grpc`.

gRPC comes with three Transport implementations:

1. The Netty-based transport is the main transport implementation based on
   [Netty](https://netty.io). It is for both the client and the server.
2. The OkHttp-based transport is a lightweight transport based on
   [OkHttp](https://square.github.io/okhttp/). It is mainly for use on Android
   and is for client only.
3. The in-process transport is for when a server is in the same process as the
   client. It is useful for testing, while also being safe for production use.
