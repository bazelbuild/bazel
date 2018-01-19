![KryoNet](https://raw.github.com/wiki/EsotericSoftware/kryo/images/logo.jpg)

[![Build Status](https://jenkins.inoio.de/buildStatus/icon?job=kryo)](https://jenkins.inoio.de/job/kryo/)

Kryo is a fast and efficient object graph serialization framework for Java. The goals of the project are speed, efficiency, and an easy to use API. The project is useful any time objects need to be persisted, whether to a file, database, or over the network.

Kryo can also perform automatic deep and shallow copying/cloning. This is direct copying from object to object, not object->bytes->object.

This documentation is for v2+ of Kryo. See [V1Documentation](https://github.com/EsotericSoftware/kryo/wiki/Documentation-for-Kryo-version-1.x) for v1.x.

If you are planning to use Kryo for network communication, the [KryoNet](https://github.com/EsotericSoftware/kryonet) project may prove useful.

## Contents

- [New in release 3.0.0](#new-in-release-300)
- [Installation](#installation)
 - [Integration with Maven](#integration-with-maven)
 - [Using Kryo without Maven](#using-kryo-without-maven)
- [Quickstart](#quickstart)
- [IO](#io)
- [Unsafe-based IO](#unsafe-based-io)
- [Serializers](#serializers)
- [Registration](#registration)
- [Default serializers](#default-serializers)
- [FieldSerializer](#fieldserializer)
- [KryoSerializable](#kryoserializable)
- [Class fields annotations](#class-fields-annotations)
- [Java Serialization](#using-standard-java-serialization)
- [Reading and writing](#reading-and-writing)
- [References](#references)
- [Object creation](#object-creation)
- [Copying/cloning](#copyingcloning)
- [Context](#context)
- [Compression and encryption](#compression-and-encryption)
- [Chunked encoding](#chunked-encoding)
- [Compatibility](#compatibility)
- [Interoperability](#interoperability)
- [Stack size](#stack-size)
- [Threading](#threading)
- [Pooling Kryo instances](#pooling-kryo-instances)
- [Logging](#logging)
- [Scala](#scala)
- [Objective-C](#objective-c)
- [Benchmarks](#benchmarks)
- [Projects using Kryo](#projects-using-kryo)
- [Contact / Mailing list](#contact--mailing-list)

## New in release 3.0.0

The 3.0.0 release fixes many reported issues and improves stability and performance. The maven groupId is changed from `com.esotericsoftware.kryo` to `com.esotericsoftware`. The Unsafe-based IO serialization format was changed and is incompatible with previous versions (therefore the new major version), the standard serialization format is still compatible.

See [ChangeLog](https://github.com/EsotericSoftware/kryo/blob/master/CHANGES.md) for more details about this release.

## Installation

Kryo JARs are available on the [releases page](https://github.com/EsotericSoftware/kryo/releases) and at [Maven Central](http://search.maven.org/#browse|1975274176). Latest snapshots of Kryo including snapshot builds of master are in the [Sonatype Repository](https://oss.sonatype.org/content/repositories/snapshots/com/esotericsoftware/kryo/kryo).

### Integration with Maven

To use the official release of Kryo, please use the following snippet in your pom.xml

```xml
    <dependency>
        <groupId>com.esotericsoftware</groupId>
        <artifactId>kryo</artifactId>
        <version>3.0.1</version>
    </dependency>
```

If you experience issues because you already have a different version of asm in your classpath, you can use the kryo-shaded jar which has its version of asm included, relocated in a different package:

```xml
    <dependency>
        <groupId>com.esotericsoftware</groupId>
        <artifactId>kryo-shaded</artifactId>
        <version>3.0.1</version>
    </dependency>
```

If you want to test the latest snapshot of Kryo, please use the following snippet in your pom.xml

```xml
    <repository>
       <id>sonatype-snapshots</id>
       <name>sonatype snapshots repo</name>
       <url>https://oss.sonatype.org/content/repositories/snapshots</url>
    </repository>
    
    <dependency>
       <groupId>com.esotericsoftware</groupId>
       <artifactId>kryo</artifactId>
        <version>3.0.1-SNAPSHOT</version>
    </dependency>
```

### Using Kryo without Maven

If you use Kryo without Maven, be aware that Kryo jar file has a couple of external dependencies, whose JARs you need to add to your classpath as well. These dependencies are [MinLog logging library](https://github.com/EsotericSoftware/minlog/) and [Objenesis library](https://code.google.com/p/objenesis/).


## Quickstart

Jumping ahead to show how the library is used:

```java
    Kryo kryo = new Kryo();
    // ...
    Output output = new Output(new FileOutputStream("file.bin"));
    SomeClass someObject = ...
    kryo.writeObject(output, someObject);
    output.close();
    // ...
    Input input = new Input(new FileInputStream("file.bin"));
    SomeClass someObject = kryo.readObject(input, SomeClass.class);
    input.close();
```

The Kryo class orchestrates serialization. The Output and Input classes handle buffering bytes and optionally flushing to a stream.

The rest of this document details how this works and advanced usage of the library.

## IO

The Output class is an OutputStream that writes data to a byte array buffer. This buffer can be obtained and used directly, if a byte array is desired. If the Output is given an OutputStream, it will flush the bytes to the stream when the buffer becomes full. Output has many methods for efficiently writing primitives and strings to bytes. It provides functionality similar to DataOutputStream, BufferedOutputStream, FilterOutputStream, and ByteArrayOutputStream.

Because Output buffers when writing to an OutputStream, be sure to call `flush()` or `close()` after writing is complete so the buffered bytes are written to the underlying stream.

The Input class is an InputStream that reads data from a byte array buffer. This buffer can be set directly, if reading from a byte array is desired. If the Input is given an InputStream, it will fill the buffer from the stream when the buffer is exhausted. Input has many methods for efficiently reading primitives and strings from bytes. It provides functionality similar to DataInputStream, BufferedInputStream, FilterInputStream, and ByteArrayInputStream.

To read from a source or write to a target other than a byte array, simply provide the appropriate InputStream or OutputStream.

## Unsafe-based IO

Kryo provides additional IO classes, which are based on the functionalities exposed by the sun.misc.Unsafe class. These classes are UnsafeInput, UnsafeOutput. They are derived from Kryo's Input and Output classes and therefore can be used as a drop-in replacement on those platforms, which properly support sun.misc.Unsafe.

For the case you need to serialize to or deserialize from direct-memory ByteBuffers or even off-heap memory, there are two dedicated classes UnsafeMemoryInput and UnsafeMemoryOutput whose instances can be used for this purpose instead of the usual Input and Output classes.

Using Unsafe-based IO may result in a quite significant performance boost (sometimes up-to an order of magnitude), depending on your application. In particular, it helps a lot when serializing large primitive arrays as part of your object graphs.

### ** DISCLAIMER ABOUT USING UNSAFE-BASED IO **

*Unsafe-based IO is not 100% compatible with Kryo's Input and Output streams when it comes to the binary format of serialized data.* 

This means that data written by Unsafe-based output streams can be read only by Unsafe-based input streams, but not by usual Input streams. The same applies on the opposite direction: data written by usual Output streams cannot be correctly read by Unsafe-based input streams.

It should be safe to use Unsafe IO streams as long as both serialization and deserialization are using them and are executed on the same processor architecture (more precisely, if the endianness and internal representation of native integer and floating point types is the same).

Unsafe IO was extensively tested on X86 hardware. Other processor architectures are not tested to the same extent. For example, there were some bug reports from users trying to use it on SPARC-based platforms. 

## Serializers

Kryo is a serialization framework. It doesn't enforce a schema or care what data is written or read. This is left to the serializers themselves. Serializers are provided by default to read and write data in various ways. If these don't meet particular needs, they can be replaced in part or in whole. The provided serializers can read and write most objects but, if necessary, writing a new serializer is easy. The Serializer abstract class defines methods to go from objects to bytes and bytes to objects.

```java
    public class ColorSerializer extends Serializer<Color> {
    	public void write (Kryo kryo, Output output, Color object) {
    		output.writeInt(object.getRGB());
    	}
    
    	public Color read (Kryo kryo, Input input, Class<T> type) {
    		return new Color(input.readInt(), true);
    	}
    }
```

Serializer has two methods that can be implemented. `write()` writes the object as bytes. `read()` creates a new instance of the object and reads from the input to populate it.

The Kryo instance can be used to write and read nested objects. If Kryo is used to read a nested object in `read()` then `kryo.reference()` must first be called with the parent object if it is possible for the nested object to reference the parent object. It is unnecessary to call `kryo.reference()` if the nested objects can't possibly reference the parent object, Kryo is not being used for nested objects, or references are not being used. If nested objects can use the same serializer, the serializer must be reentrant.

Code should not make use of serializers directly, instead the Kryo read and write methods should be used. This allows Kryo to orchestrate serialization and handle features such as references and null objects.

By default, serializers do not need to handle the object being null. The Kryo framework will write a byte as needed denoting null or not null. If a serializer wants to be more efficient and handle nulls itself, it can call `Serializer#setAcceptsNull(true)`. This can also be used to avoid writing the null denoting byte when it is known that all instances of a type will never be null.

## Registration

When Kryo writes out an instance of an object, first it may need to write out something that identifies the object's class. By default, the fully qualified class name is written, then the bytes for the object. Subsequent appearances of that object type within the same object graph are written using a variable length int. Writing the class name is somewhat inefficient, so classes can be registered beforehand:

```java
    Kryo kryo = new Kryo();
    kryo.register(SomeClass.class);
    // ...
    Output output = ...
    SomeClass someObject = ...
    kryo.writeObject(output, someObject);
```

Here SomeClass is registered with Kryo, which associates the class with an int ID. When Kryo writes out an instance of SomeClass, it will write out this int ID. This is more efficient than writing out the class name, but requires the classes that will be serialized to be known up front. During deserialization, the registered classes must have the exact same IDs they had during serialization. The register method shown above assigns the next available, lowest integer ID, which means the order classes are registered is important. The ID can also be specified explicitly to make order unimportant:

```java
    Kryo kryo = new Kryo();
    kryo.register(SomeClass.class, 0);
    kryo.register(AnotherClass.class, 1);
    kryo.register(YetAnotherClass.class, 2);
```

The IDs are written most efficiently when they are small, positive integers. Negative IDs are not serialized efficiently. -1 and -2 are reserved.

Use of registered and unregistered classes can be mixed. All primitives, primitive wrappers, and String are registered by default.

Kryo#setRegistrationRequired can be set to true to throw an exception when any unregistered class is encountered. This prevents an application from accidentally using class name strings.

If using unregistered classes, short package names could be considered.

## Default serializers

After writing the class identifier, Kryo uses a serializer to write the object's bytes. When a class is registered, a serializer instance can be specified:

```java
    Kryo kryo = new Kryo();
    kryo.register(SomeClass.class, new SomeSerializer());
    kryo.register(AnotherClass.class, new AnotherSerializer());
```

If a class is not registered or no serializer is specified, a serializer is chosen automatically from a list of "default serializers" that maps a class to a serializer. The following classes have a default serializer set by default:


<table>
  <tr><td>boolean</td><td>Boolean</td><td>byte</td><td>Byte</td><td>char</td></tr>
  <tr><td>Character</td><td>short</td><td>Short</td><td>int</td><td>Integer</td></tr>
  <tr><td>long</td><td>Long</td><td>float</td><td>Float</td><td>double</td></tr>
  <tr><td>Double</td><td>byte[]</td><td>String</td><td>BigInteger</td><td>BigDecimal</td></tr>
  <tr><td>Collection</td><td>Date</td><td>Collections.emptyList</td><td>Collections.singleton</td><td>Map</td></tr>
  <tr><td>StringBuilder</td><td>TreeMap</td><td>Collections.emptyMap</td><td>Collections.emptySet</td><td>KryoSerializable</td></tr>
  <tr><td>StringBuffer</td><td>Class</td><td>Collections.singletonList</td><td>Collections.singletonMap</td><td>Currency</td></tr>
  <tr><td>Calendar</td><td>TimeZone</td><td>Enum</td><td>EnumSet</td></tr>
</table>


Additional default serializers can be added:

```java
    Kryo kryo = new Kryo();
    kryo.addDefaultSerializer(SomeClass.class, SomeSerializer.class);
    // ...
    Output output = ...
    SomeClass someObject = ...
    kryo.writeObject(output, someObject);
```

A class can also use the DefaultSerializer annotation:

```java
    @DefaultSerializer(SomeClassSerializer.class)
    public class SomeClass {
       // ...
    }
```

If no default serializers match a class, then by default [FieldSerializer](#FieldSerializer) is used. This can also be changed:

    Kryo kryo = new Kryo();
    kryo.setDefaultSerializer(AnotherGenericSerializer.class);

Some serializers allow extra information to be provided so that the number of bytes output can be reduced:

```java
    Kryo kryo = new Kryo();
    FieldSerializer someClassSerializer = new FieldSerializer(kryo, SomeClass.class);
    CollectionSerializer listSerializer = new CollectionSerializer();
    listSerializer.setElementClass(String.class);
    listSerializer.setElementsCanBeNull(false);
    someClassSerializer.getField("list").setClass(LinkedList.class, listSerializer);
    kryo.register(SomeClass.class, someClassSerializer);
    // ...
    SomeClass someObject = ...
    someObject.list = new LinkedList();
    someObject.list.add("thishitis");
    someObject.list.add("bananas");
    kryo.writeObject(output, someObject);
```

In this example, FieldSerializer will be used for SomeClass. FieldSerializer is configured so the "list" field will always be a LinkedList and will use the specified CollectionSerializer. The CollectionSerializer is configured so each element will be a String and none of the elements will be null. This allows the serializer to be more efficient. In this case, 2 to 3 bytes are saved per element in the list.

## FieldSerializer

By default, most classes will end up using FieldSerializer. It essentially does what hand written serialization would, but does it automatically. FieldSerializer does direct assignment to the object's fields. If the fields are public, protected, or default access (package private) and not marked as final, bytecode generation is used for maximum speed (see [ReflectASM](https://github.com/EsotericSoftware/reflectasm)). For private fields, setAccessible and cached reflection is used, which is still quite fast.

Other general purpose serializes are provided, such as BeanSerializer, TaggedFieldSerializer, CompatibleFieldSerializer, and VersionFieldSerializer. Additional serializers are available in a separate project on github, [kryo-serializers](https://github.com/magro/kryo-serializers).

## KryoSerializable

While FieldSerializer is ideal for most classes, sometimes it is convenient for a class to do its own serialization. This can be done by implementing KryoSerializable interface (similar to the java.io.Externalizable interface in the JDK).

```java

    public class SomeClass implements KryoSerializable {
       // ...
    
       public void write (Kryo kryo, Output output) {
          // ...
       }
    
       public void read (Kryo kryo, Input input) {
          // ...
       }
    }
```


## Using standard Java Serialization

While very rare, some classes cannot be serialized by Kryo. In such situations it is possible to use a fallback solution provided by Kryo's JavaSerializer and use the standard Java Serialization instead. This approach would be as slow as usual Java serialization, but would make your class serialize as long as Java serialization is able to serialize it. Of course, your classs should implement the `Serializable` or `Externalizable` interface as it is required by usual Java serialization.

If your class impements Java's `Serializable` interface, then you may want to use Kryo's dedicated `JavaSerializer` serializer for it:

```java
    kryo.register(SomeClass.class, new JavaSerializer());
```


If your class impements Java's `Externalizable` interface, then you may want to use Kryo's dedicated `ExternalizableSerializer` serializer for it:

```java
    kryo.register(SomeClass.class, new ExternalizableSerializer());
```


## Class fields annotations

Typically, when FieldSerializer is used it is able to automatically guess which serializer should be used for each field of a class. But in certain situations you may want to change a default behavior and customize the way how this field is serialized.

Kryo provides a set of annotations that can be used exactly for this purpose. `@Bind` can be used for any field, `@CollectionBind` for fields whose type is a collection and `@MapBind` for fields whose type is a map:

```java

    public class SomeClass {
       // Use a StringSerializer for this field
       @Bind(StringSerializer.class) 
       Object stringField;
       
       // Use a MapSerializer for this field. Keys should be serialized
       // using a StringSerializer, whereas values should be serialized
       // using IntArraySerializer
       @BindMap(
     			valueSerializer = IntArraySerializer.class, 
     			keySerializer = StringSerializer.class, 
     			valueClass = int[].class, 
     			keyClass = String.class, 
     			keysCanBeNull = false) 
       Map map;
       
       // Use a CollectionSerializer for this field. Elements should be serialized
       // using LongArraySerializer
       @BindCollection(
     			elementSerializer = LongArraySerializer.class,
     			elementClass = long[].class, 
     			elementsCanBeNull = false) 
       Collection collection;
       
       // ...
    }
```


## Reading and writing

Kryo has three sets of methods for reading and writing objects.

If the concrete class of the object is not known and the object could be null:

```java
    kryo.writeClassAndObject(output, object);
    // ...
    Object object = kryo.readClassAndObject(input);
    if (object instanceof SomeClass) {
       // ...
    }
```

If the class is known and the object could be null:

```java
    kryo.writeObjectOrNull(output, someObject);
    // ...
    SomeClass someObject = kryo.readObjectOrNull(input, SomeClass.class);
```

If the class is known and the object cannot be null:

```java
    kryo.writeObject(output, someObject);
    // ...
    SomeClass someObject = kryo.readObject(input, SomeClass.class);
```

## References

By default, each appearance of an object in the graph after the first is stored as an integer ordinal. This allows multiple references to the same object and cyclic graphs to be serialized. This has a small amount of overhead and can be disabled to save space if it is not needed:

```java
    Kryo kryo = new Kryo();
    kryo.setReferences(false);
    // ...
```

When writing serializers that use Kryo for nested objects, `kryo.reference()` must be called in `read()`. See [Serializers](#serializers) for more information.

## Object creation

Serializers for a specific type use Java code to create a new instance of that type. Serializers such as FieldSerializer are generic and must handle creating a new instance of any class. By default, if a class has a zero argument constructor then it is invoked via [ReflectASM](http://code.google.com/p/reflectasm/) or reflection, otherwise an exception is thrown. If the zero argument constructor is private, an attempt is made to access it via reflection using setAccessible. If this is acceptable, a private zero argument constructor is a good way to allow Kryo to create instances of a class without affecting the public API.

When ReflectASM or reflection cannot be used, Kryo can be configured to use an InstantiatorStrategy to handle creating instances of a class. [Objenesis](https://code.google.com/p/objenesis/) provides StdInstantiatorStrategy which uses JVM specific APIs to create an instance of a class without calling any constructor at all. While this works on many JVMs, a zero argument is generally more portable.

```java
    kryo.setInstantiatorStrategy(new StdInstantiatorStrategy());
```

Note that classes must be designed to be created in this way. If a class expects its constructor to be called, it may be in an uninitialized state when created through this mechanism.

In many situations, you may want to have a strategy, where Kryo first tries to find and use a no-arg constructor and if it fails to do so, it should try to use `StdInstantiatorStrategy` as a fallback, because this one does not invoke any constructor at all. This is actually the default configuration and could be expressed like this:

```java
kryo.setInstantiatorStrategy(new DefaultInstantiatorStrategy(new StdInstantiatorStrategy()));
```

Objenesis can also create new objects using Java's built-in serialization mechanism. Using this, the class must implement java.io.Serializable and the first zero argument constructor in a super class is invoked.

```java
    kryo.setInstantiatorStrategy(new SerializingInstantiatorStrategy());
```

You may also write your own InstantiatorStrategy.

To customize only how a specific type is created, an ObjectInstantiator can be set. This will override ReflectASM, reflection, and the InstantiatorStrategy.

```java
    Registration registration = kryo.register(SomeClass.class);
    registration.setObjectInstantiator(...);
```

Alternatively, some serializers provide methods that can be overridden to customize object creation.

```java
    kryo.register(SomeClass.class, new FieldSerializer(kryo, SomeClass.class) {
       public Object create (Kryo kryo, Input input, Class type) {
          return new SomeClass("some constructor arguments", 1234);
       }
    });
```

## Copying/cloning

A serialization library needs special knowledge on how to create new instances, get and set values, navigate object graphs, etc. This is nearly everything needed to support copying objects, so it makes sense for Kryo to support automatically making deep and shallow copies of objects. Note Kryo's copying does not serialize to bytes and back, it uses direct assignment.

```java
    Kryo kryo = new Kryo();
    SomeClass someObject = ...
    SomeClass copy1 = kryo.copy(someObject);
    SomeClass copy2 = kryo.copyShallow(someObject);
```

The Serializer class has a `copy` method that does the work. These methods can be ignored when implementing application specific serializers if the copying functionality will not be used. All serializers provided with Kryo support copying. Multiple references to the same object and circular references are handled by the framework automatically.

Similar to the `read()` Serializer method, `kryo.reference()` must be called before Kryo can be used to copy child objects. See [Serializers](#Serializers) for more information.

Similar to KryoSerializable, classes can implement KryoCopyable to do their own copying:

```java
    public class SomeClass implements KryoCopyable<SomeClass> {
       // ...
    
       public SomeClass copy (Kryo kryo) {
          // Create new instance and copy values from this instance.
       }
    }
```

## Context

Kryo has two context methods. `getContext()` returns a map for storing user data. Because the Kryo instance is available to all serializers, this data is readily available. `getGraphContext()` is similar, but is cleared after each object graph is serialized or deserialized. This makes it easy to manage per object graph state.

## Compression and encryption

Kryo supports streams, so it is trivial to use compression or encryption on all of the serialized bytes:

```java
    OutputStream outputStream = new DeflaterOutputStream(new FileOutputStream("file.bin"));
    Output output = new Output(outputStream);
    Kryo kryo = new Kryo();
    kryo.writeObject(output, object);
    output.close();
```

If needed, a serializer can be used to compress or encrypt the bytes for only a subset of the bytes for an object graph. For example, see DeflateSerializer or BlowfishSerializer. These serializers wrap another serializer and encode and decode the bytes.

## Chunked encoding

Sometimes it is useful to write the length of some data, then the data. If the length of the data is not known ahead of time, all the data would need to be buffered to determine its length, then the length can be written, then the data. This buffering prevents streaming and potentially requires a very large buffer, which is not ideal.

Chunked encoding solves this by using a small buffer. When the buffer is full, its length is written, then the data. This is one chunk of data. The buffer is cleared and this continues until there is no more data to write. A chunk with a length of zero denotes the end of the chunks.

Kryo provides classes for easy chunked encoding. OutputChunked is used to write chunked data. It extends Output, so has all the convenient methods to write data. When the OutputChunked buffer is full, it flushes the chunk to the wrapped OutputStream. The `endChunks()` method is used to mark the end of a set of chunks.

```java
    OutputStream outputStream = new FileOutputStream("file.bin");
    OutputChunked output = new OutputChunked(outputStream, 1024);
    // Write data to output...
    output.endChunks();
    // Write more data to output...
    output.endChunks();
    // Write even more data to output...
    output.close();
```

To read the chunked data, InputChunked is used. It extends Input, so has all the convenient methods to read data. When reading, InputChunked will appear to hit the end of the data when it reaches the end of a set of chunks. The `nextChunks()` method advances to the next set of chunks, even if not all the data has been read from the current set of chunks.

```java
    InputStream outputStream = new FileInputStream("file.bin");
    InputChunked input = new InputChunked(inputStream, 1024);
    // Read data from first set of chunks...
    input.nextChunks();
    // Read data from second set of chunks...
    input.nextChunks();
    // Read data from third set of chunks...
    input.close();
```

## Compatibility

For some needs, especially long term storage of serialized bytes, it can be important how serialization handles changes to classes. This is known as forward (reading bytes serialized by newer classes) and backward (reading bytes serialized by older classes) compatibility.

FieldSerializer is the most commonly used serializer. It is generic and can serialize most classes without any configuration. It is efficient and writes only the field data, without any extra information. It does not support adding, removing, or changing the type of fields without invalidating previously serialized bytes. This can be acceptable in many situations, such as when sending data over a network, but may not be a good choice for long term data storage because the Java classes cannot evolve. Because FieldSerializer attempts to read and write non-public fields by default, it is important to evaluate each class that will be serialized.

When no serializer is specified, FieldSerializer is used by default. If necessary, an alternate generic serializer can be used:

```java
kryo.setDefaultSerializer(TaggedFieldSerializer.class);
```

BeanSerializer is very similar to FieldSerializer, except it uses bean getter and setter methods rather than direct field access. This slightly slower, but may be safer because it uses the public API to configure the object.

VersionFieldSerializer extends FieldSerializer and allows fields to have a `@Since(int)` annotation to indicate the version they were added. For a particular field, the value in `@Since` should never change once created. This is less flexible than FieldSerializer, which can handle most classes without needing annotations, but it provides backward compatibility. This means that new fields can be added, but removing, renaming or changing the type of any field will invalidate previous serialized bytes. VersionFieldSerializer has very little overhead (a single additional varint) compared to FieldSerializer.

TaggedFieldSerializer extends FieldSerializer to only serialize fields that have a `@Tag(int)` annotation, providing backward compatibility so new fields can be added. TaggedFieldSerializer has two advantages over VersionFieldSerializer: 1) fields can be renamed and 2) fields marked with the `@Deprecated` annotation will be ignored when reading old bytes and won't be written to new bytes. Deprecation effectively removes the field from serialization, though the field and `@Tag` annotation must remain in the class. Deprecated fields can optionally be made private and/or renamed so they don't clutter the class (eg, `ignored`, `ignored2`). For these reasons, TaggedFieldSerializer generally provides more flexibility for classes to evolve. The downside is that it has a small amount of additional overhead compared to VersionFieldSerializer (an additional varint per field).

CompatibleFieldSerializer extends FieldSerializer to provide both forward and backward compatibility, meaning fields can be added or removed without invalidating previously serialized bytes. Changing the type of a field is not supported. Like FieldSerializer, it can serialize most classes without needing annotations. The forward and backward compatibility comes at a cost: the first time the class is encountered in the serialized bytes, a simple schema is written containing the field name strings. Also, during serialization and deserialization buffers are allocated to perform chunked encoding. This is what enables CompatibleFieldSerializer to skip bytes for fields it does not know about. When Kryo is configured to use references, there can be a [problem](https://github.com/EsotericSoftware/kryo/issues/286#issuecomment-74870545) with CompatibleFieldSerializer if a field is removed.

Additional serializers can easily be developed for forward and backward compatibility, such as a serializer that uses an external, hand written schema.

## Interoperability

The Kryo serializers provided by default assume that Java will be used for deserialization, so they do not explicitly define the format that is written. Serializers could be written using a standardized format that is more easily read by another language, but this is not provided by default.

## Stack size

The serializers Kryo provides use the call stack when serializing nested objects. Kryo does minimize stack calls, but for extremely deep object graphs, a stack overflow can occur. This is a common issue for most serialization libraries, including the built-in Java serialization. The stack size can be increased using `-Xss`, but note that this is for all threads. Large stack sizes in a JVM with many threads may use a large amount of memory.

## Threading

**Kryo is not thread safe. Each thread should have its own Kryo, Input, and Output instances. Also, the byte[] Input uses may be modified and then returned to its original state during deserialization, so the same byte[] "should not be used concurrently in separate threads**.

## Pooling Kryo instances

Because the creation/initialization of `Kryo` instances is rather expensive, in a multithreaded scenario you should pool `Kryo` instances.
A very simple solution is to bind `Kryo` instances to Threads using `ThreadLocal`, like this:

```java
// Setup ThreadLocal of Kryo instances
private ThreadLocal<Kryo> kryos = new ThreadLocal<Kryo>() {
	protected Kryo initialValue() {
		Kryo kryo = new Kryo();
		// configure kryo instance, customize settings
		return kryo;
	};
};

// Somewhere else, use Kryo
Kryo k = kryos.get();
...
```

Alternatively you may want to use the `KryoPool` provided by kryo. The `KryoPool` allows to keep references to `Kryo` instances
using `SoftReference`s, so that `Kryo` instances can be GC'ed when the JVM starts to run out of memory
(of course you could use `ThreadLocal` with `SoftReference`s as well).

Here's an example that shows how to use the `KryoPool`:

```java
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.pool.*;

KryoFactory factory = new KryoFactory() {
  public Kryo create () {
    Kryo kryo = new Kryo();
    // configure kryo instance, customize settings
    return kryo;
  }
};
// Build pool with SoftReferences enabled (optional)
KryoPool pool = new KryoPool.Builder(factory).softReferences().build();
Kryo kryo = pool.borrow();
// do s.th. with kryo here, and afterwards release it
pool.release(kryo);

// or use a callback to work with kryo - no need to borrow/release,
// that's done by `run`.
String value = pool.run(new KryoCallback() {
  public String execute(Kryo kryo) {
    return kryo.readObject(input, String.class);
  }
});
```

## Logging

Kryo makes use of the low overhead, lightweight [MinLog logging library](http://code.google.com/p/minlog/). The logging level can be set by one of the following methods:

```java
    Log.ERROR();
    Log.WARN();
    Log.INFO();
    Log.DEBUG();
    Log.TRACE();
```

Kryo does no logging at `INFO` (the default) and above levels. `DEBUG` is convenient to use during development. `TRACE` is good to use when debugging a specific problem, but generally outputs too much information to leave on.

MinLog supports a fixed logging level, which causes javac to remove logging statements below that level at compile time. In the Kryo distribution ZIP, the "debug" JARs have logging enabled. The "production" JARs use a fixed logging level of `NONE`, which means all logging code has been removed.

## Scala

See the following projects which provide serializers for Scala classes:

- [Twitter's Chill](https://github.com/twitter/chill) (Kryo serializers for Scala)
- [akka-kryo-serialization](https://github.com/romix/akka-kryo-serialization) (Kryo serializers for Scala and Akka)
- [Twitter's Scalding](https://github.com/twitter/scalding) (Scala API for Cascading)
- [Kryo Serializers](https://github.com/magro/kryo-serializers) (Additional serializers for Java)

## Clojure

- [Carbonite](https://github.com/sritchie/carbonite) (Kryo serializers for Clojure)

## Objective-C

See the following project which is an Objective-C port of Kryo:
- [kryococoa](https://github.com/Feuerwerk/kryococoa)

## Benchmarks

Kryo can be compared to many other serialization libraries in the [JVM Serializers](https://github.com/eishay/jvm-serializers/wiki) project. It is difficult to thoroughly compare serialization libraries using a benchmark. They often have different goals and may excel at solving completely different problems. To understand these benchmarks, the code being run and data being serialized should be analyzed and contrasted with your specific needs. Some serializers are highly optimized and use pages of code, others use only a few lines. This is good to show what is possible, but may not be practical for many situations.

"kryo" is typical Kryo usage, classes are registered and serialization is done automatically. "kryo-opt" shows how serializers can be configured to reduce the size for the specific data being serialized, but serialization is still done automatically. "kryo-manual" shows how hand written serialization code can be used to optimize for both size and speed while still leveraging Kryo for most of the work.

## Projects using Kryo

There are a number of projects using Kryo. A few are listed below. Please post a message to the [mailing list](https://groups.google.com/forum/#!forum/kryo-users) if you'd like your project included here.

- [KryoNet](http://code.google.com/p/kryonet/) (NIO networking)
- [Twitter's Scalding](https://github.com/twitter/scalding) (Scala API for Cascading)
- [Twitter's Chill](https://github.com/twitter/chill) (Kryo serializers for Scala)
- [Apache Hive](http://hive.apache.org/) (query plan serialization)
- [DataNucleus](https://github.com/datanucleus/type-converter-kryo) (JDO/JPA persistence framework)
- [CloudPelican](http://www.cloudpelican.com/)
- [Yahoo's S4](http://www.s4.io/) (distributed stream computing)
- [Storm](https://github.com/nathanmarz/storm/wiki/Serialization) (distributed realtime computation system, in turn used by [many others](https://github.com/nathanmarz/storm/wiki/Powered-By))
- [Cascalog](https://github.com/nathanmarz/cascalog) (Clojure/Java data processing and querying [details](https://groups.google.com/d/msg/cascalog-user/qgwO2vbkRa0/UeClnLL5OsgJ))
- [memcached-session-manager](https://code.google.com/p/memcached-session-manager/) (Tomcat high-availability sessions)
- [Mobility-RPC](http://code.google.com/p/mobility-rpc/) (RPC enabling distributed applications)
- [akka-kryo-serialization](https://github.com/romix/akka-kryo-serialization) (Kryo serializers for Akka)
- [Groupon](https://code.google.com/p/kryo/issues/detail?id=67)
- [Jive](http://www.jivesoftware.com/jivespace/blogs/jivespace/2010/07/29/the-jive-sbs-cache-redesign-part-3)
- [DestroyAllHumans](https://code.google.com/p/destroyallhumans/) (controls a [robot](http://www.youtube.com/watch?v=ZeZ3R38d3Cg)!)
- [kryo-serializers](https://github.com/magro/kryo-serializers) (additional serializers)

## Contact / Mailing list

You can use the [kryo mailing list](https://groups.google.com/forum/#!forum/kryo-users) for questions/discussions/support.
