![](https://raw.github.com/wiki/EsotericSoftware/reflectasm/images/logo.png)

Please use the [ReflectASM discussion group](http://groups.google.com/group/reflectasm-users) for support.

## Overview

ReflectASM is a very small Java library that provides high performance reflection by using code generation. An access class is generated to set/get fields, call methods, or create a new instance. The access class uses bytecode rather than Java's reflection, so it is much faster. It can also access primitive fields via bytecode to avoid boxing.

## Performance

![](http://chart.apis.google.com/chart?chma=100&chtt=Field%20Set/Get&chs=700x62&chd=t:1402081,11339107&chds=0,11339107&chxl=0:|Java%20Reflection|FieldAccess&cht=bhg&chbh=10&chxt=y&chco=6600FF)

![](http://chart.apis.google.com/chart?chma=100&chtt=Method%20Call&chs=700x62&chd=t:97390,208750&chds=0,208750&chxl=0:|Java%20Reflection|MethodAccess&cht=bhg&chbh=10&chxt=y&chco=6600AA)

![](http://chart.apis.google.com/chart?chma=100&chtt=Constructor&chs=700x62&chd=t:2853063,5828993&chds=0,5828993&chxl=0:|Java%20Reflection|ConstructorAccess&cht=bhg&chbh=10&chxt=y&chco=660066)

The source code for these benchmarks is included in the project. The above charts were generated on Oracle's Java 7u3, server VM.

## Usage

Method reflection with ReflectASM:

```java
    SomeClass someObject = ...
    MethodAccess access = MethodAccess.get(SomeClass.class);
    access.invoke(someObject, "setName", "Awesome McLovin");
    String name = (String)access.invoke(someObject, "getName");
```

Field reflection with ReflectASM:

```java
    SomeClass someObject = ...
    FieldAccess access = FieldAccess.get(SomeClass.class);
    access.set(someObject, "name", "Awesome McLovin");
    String name = (String)access.get(someObject, "name");
```

Constructor reflection with ReflectASM:

```java
    ConstructorAccess<SomeClass> access = ConstructorAccess.get(SomeClass.class);
    SomeClass someObject = access.newInstance();
```

## Avoiding Name Lookup

For maximum performance when methods or fields are accessed repeatedly, the method or field index should be used instead of the name:

```java
    SomeClass someObject = ...
    MethodAccess access = MethodAccess.get(SomeClass.class);
    int addNameIndex = access.getIndex("addName");
    for (String name : names)
        access.invoke(someObject, addNameIndex, "Awesome McLovin");
```

Iterate all fields:

```java
    FieldAccess access = FieldAccess.get(SomeClass.class);
    for(int i = 0, n = access.getFieldCount(); i < n; i++) {
        access.set(instanceObject, i, valueToPut);              
    }
 }

```

## Visibility

ReflectASM can always access public members. An attempt is made to define access classes in the same classloader (using setAccessible) and package as the accessed class. If the security manager allows setAccessible to succeed, then protected and default access (package private) members can be accessed. If setAccessible fails, no exception is thrown, but only public members can be accessed. Private members can never be accessed.

## Exceptions

Stack traces when using ReflectASM are a bit cleaner. Here is Java's reflection calling a method that throws a RuntimeException:

    Exception in thread "main" java.lang.reflect.InvocationTargetException
    	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:39)
    	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:25)
    	at java.lang.reflect.Method.invoke(Method.java:597)
    	at com.example.SomeCallingCode.doit(SomeCallingCode.java:22)
    Caused by: java.lang.RuntimeException
    	at com.example.SomeClass.someMethod(SomeClass.java:48)
    	... 5 more

Here is the same but when ReflectASM is used:

    Exception in thread "main" java.lang.RuntimeException
    	at com.example.SomeClass.someMethod(SomeClass.java:48)
    	at com.example.SomeClassMethodAccess.invoke(Unknown Source)
    	at com.example.SomeCallingCode.doit(SomeCallingCode.java:22)

If ReflectASM is used to invoke code that throws a checked exception, the checked exception is thrown. Because it is a compilation error to use try/catch with a checked exception around code that doesn't declare that exception as being thrown, you must catch Exception if you care about catching a checked exception in code you invoke with ReflectASM.
