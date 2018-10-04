// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.android.desugar.testdata.java8;

/**
 * The base interface, which is generic, and has two default methods. These two default methods will
 * introduce bridge methods in the child-interfaces
 */
interface GenericInterfaceWithDefaultMethod<T extends Number> {
  default T copy(T t) {
    return t;
  }

  default Number getNumber() {
    return 1;
  }
}

/** This interface generate two additional bridge methods */
interface InterfaceWithDefaultAndBridgeMethods extends GenericInterfaceWithDefaultMethod<Integer> {
  @Override
  default Integer copy(Integer t) {
    return GenericInterfaceWithDefaultMethod.super.copy(t);
  }

  @Override
  default Double getNumber() {
    return 2.3d;
  }
}

/** A class implementing the interface. */
class ClassWithDefaultAndBridgeMethods implements InterfaceWithDefaultAndBridgeMethods {}

/** The client class that uses the interfaces and the class that implements the interfaces. */
public class DefaultInterfaceWithBridges {
  private final ClassWithDefaultAndBridgeMethods c = new ClassWithDefaultAndBridgeMethods();

  public Integer copy(Integer i) {
    return c.copy(i);
  }

  @SuppressWarnings({"rawtypes", "unchecked"})
  public Number copy(Number n) {
    return ((GenericInterfaceWithDefaultMethod) c).copy(n);
  }

  public Number getNumber() {
    return ((GenericInterfaceWithDefaultMethod) c).getNumber();
  }

  public Double getDouble() {
    return c.getNumber();
  }
}
