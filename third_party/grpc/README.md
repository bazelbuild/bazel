gRPC-Java - An RPC library and framework
========================================

gRPC-Java works with JDK 6. TLS usage typically requires using Java 8, or Play
Services Dynamic Security Provider on Android. Please see the [Security
Readme](SECURITY.md).

<table>
  <tr>
    <td><b>Homepage:</b></td>
    <td><a href="http://www.grpc.io/">www.grpc.io</a></td>
  </tr>
  <tr>
    <td><b>Mailing List:</b></td>
    <td><a href="https://groups.google.com/forum/#!forum/grpc-io">grpc-io@googlegroups.com</a></td>
  </tr>
</table>

[![Build Status](https://travis-ci.org/grpc/grpc-java.svg?branch=master)](https://travis-ci.org/grpc/grpc-java)
[![Coverage Status](https://coveralls.io/repos/grpc/grpc-java/badge.svg?branch=master&service=github)](https://coveralls.io/github/grpc/grpc-java?branch=master)

Download
--------

Download [the JAR][]. Or for Maven, add to your `pom.xml`:
```xml
<dependency>
  <groupId>io.grpc</groupId>
  <artifactId>grpc-all</artifactId>
  <version>0.13.2</version>
</dependency>
```

Or for Gradle, add to your dependencies:
```gradle
compile 'io.grpc:grpc-all:0.13.2'
```

For Android client, you only need to depend on the needed sub-projects, such as:
```gradle
compile 'io.grpc:grpc-okhttp:0.13.2'
compile 'io.grpc:grpc-protobuf-nano:0.13.2'
compile 'io.grpc:grpc-stub:0.13.2'
```

[the JAR]: https://search.maven.org/remote_content?g=io.grpc&a=grpc-all&v=0.13.2

Development snapshots are available in [Sonatypes's snapshot
repository](https://oss.sonatype.org/content/repositories/snapshots/).

For protobuf-based codegen, you can put your proto files in the `src/main/proto`
and `src/test/proto` directories along with an appropriate plugin.

For protobuf-based codegen integrated with the Maven build system, you can use
[protobuf-maven-plugin][]:
```xml
<build>
  <extensions>
    <extension>
      <groupId>kr.motd.maven</groupId>
      <artifactId>os-maven-plugin</artifactId>
      <version>1.4.1.Final</version>
    </extension>
  </extensions>
  <plugins>
    <plugin>
      <groupId>org.xolstice.maven.plugins</groupId>
      <artifactId>protobuf-maven-plugin</artifactId>
      <version>0.5.0</version>
      <configuration>
        <!--
          The version of protoc must match protobuf-java. If you don't depend on
          protobuf-java directly, you will be transitively depending on the
          protobuf-java version that grpc depends on.
        -->
        <protocArtifact>com.google.protobuf:protoc:3.0.0-beta-2:exe:${os.detected.classifier}</protocArtifact>
        <pluginId>grpc-java</pluginId>
        <pluginArtifact>io.grpc:protoc-gen-grpc-java:0.13.2:exe:${os.detected.classifier}</pluginArtifact>
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

For protobuf-based codegen integrated with the Gradle build system, you can use
[protobuf-gradle-plugin][]:
```gradle
apply plugin: 'java'
apply plugin: 'com.google.protobuf'

buildscript {
  repositories {
    mavenCentral()
  }
  dependencies {
    classpath 'com.google.protobuf:protobuf-gradle-plugin:0.7.4'
  }
}

protobuf {
  protoc {
    // The version of protoc must match protobuf-java. If you don't depend on
    // protobuf-java directly, you will be transitively depending on the
    // protobuf-java version that grpc depends on.
    artifact = "com.google.protobuf:protoc:3.0.0-beta-2"
  }
  plugins {
    grpc {
      artifact = 'io.grpc:protoc-gen-grpc-java:0.13.2'
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

How to Build
------------

If you are making changes to gRPC-Java, see the [compiling
instructions](COMPILING.md).

Navigating Around the Source
----------------------------

Here's a quick readers' guide to the code to help folks get started. At a high
level there are three distinct layers to the library: __Stub__, __Channel__ &
__Transport__.

### Stub

The Stub layer is what is exposed to most developers and provides type-safe
bindings to whatever datamodel/IDL/interface you are adapting. gRPC comes with
a [plugin](https://github.com/google/grpc-java/blob/master/compiler) to the
protocol-buffers compiler that generates Stub interfaces out of `.proto` files,
but bindings to other datamodel/IDL should be trivial to add and are welcome.

#### Key Interfaces

[Stream Observer](https://github.com/google/grpc-java/blob/master/stub/src/main/java/io/grpc/stub/StreamObserver.java)

### Channel

The Channel layer is an abstraction over Transport handling that is suitable for
interception/decoration and exposes more behavior to the application than the
Stub layer. It is intended to be easy for application frameworks to use this
layer to address cross-cutting concerns such as logging, monitoring, auth etc.
Flow-control is also exposed at this layer to allow more sophisticated
applications to interact with it directly.

#### Common

* [Metadata - headers & trailers](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/Metadata.java)
* [Status - error code namespace & handling](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/Status.java)

#### Client
* [Channel - client side binding](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/Channel.java)
* [Client Call](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/ClientCall.java)
* [Client Interceptor](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/ClientInterceptor.java)

#### Server
* [Server call handler - analog to Channel on server](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/ServerCallHandler.java)
* [Server Call](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/ServerCall.java)


### Transport

The Transport layer does the heavy lifting of putting and taking bytes off the
wire. The interfaces to it are abstract just enough to allow plugging in of
different implementations. Transports are modeled as `Stream` factories. The
variation in interface between a server Stream and a client Stream exists to
codify their differing semantics for cancellation and error reporting.

Note the transport layer API is considered internal to gRPC and has weaker API
guarantees than the core API under package `io.grpc`.

gRPC comes with three Transport implementations:

1. The [Netty-based](https://github.com/google/grpc-java/blob/master/netty)
   transport is the main transport implementation based on
   [Netty](http://netty.io). It is for both the client and the server.
2. The [OkHttp-based](https://github.com/google/grpc-java/blob/master/okhttp)
   transport is a lightweight transport based on
   [OkHttp](http://square.github.io/okhttp/). It is mainly for use on Android
   and is for client only.
3. The
   [inProcess](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/inprocess)
   transport is for when a server is in the same process as the client. It is
   useful for testing.

#### Common

* [Stream](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/internal/Stream.java)
* [Stream Listener](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/internal/StreamListener.java)

#### Client

* [Client Stream](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/internal/ClientStream.java)
* [Client Stream Listener](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/internal/ClientStreamListener.java)

#### Server

* [Server Stream](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/internal/ServerStream.java)
* [Server Stream Listener](https://github.com/google/grpc-java/blob/master/core/src/main/java/io/grpc/internal/ServerStreamListener.java)


### Examples

Tests showing how these layers are composed to execute calls using protobuf
messages can be found here
https://github.com/google/grpc-java/tree/master/interop-testing/src/main/java/io/grpc/testing/integration
