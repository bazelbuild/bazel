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
 * An interface that has a default method, and a non-empty static initializer. The initializer is
 * NOT expected to run during desugaring.
 */
public interface FunctionalInterfaceWithInitializerAndDefaultMethods {

  ClassWithInitializer CONSTANT = new ClassWithInitializer();
  boolean BOOLEAN = getFalse();
  char CHAR = "hello".charAt(0);
  byte BYTE = Byte.parseByte("0");
  short SHORT = Short.parseShort("0");
  int INT = Integer.parseInt("0");
  float FLOAT = Float.parseFloat("0");
  long LONG = Long.parseLong("0");
  double DOUBLE = Double.parseDouble("0");

  int convert();

  /**
   * The default method ensures that the static initializer of this interface will be executed when
   * the interface is loaded.
   */
  default void defaultMethod() {}

  static boolean getFalse() {
    return false;
  }

  /**
   * A class with a static initializer that has side effects (In this class, the printing to stdout)
   */
  class ClassWithInitializer {
    static {
      System.out.println("THIS STRING IS NOT EXPECTED TO APPEAR IN THE OUTPUT OF DESUGAR!!!");
    }

    /**
     * A lambda to trigger Desugar to load the interface {@link
     * FunctionalInterfaceWithInitializerAndDefaultMethods}
     */
    public FunctionalInterfaceWithInitializerAndDefaultMethods length(String s) {
      return s::length;
    }
  }
}
