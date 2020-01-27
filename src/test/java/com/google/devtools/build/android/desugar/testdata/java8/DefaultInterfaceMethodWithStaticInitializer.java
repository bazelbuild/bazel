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

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;

/**
 * Interfaces with default methods are intialized differently from those without default methods.
 * When we load such an interface, its static intializer will be executed.
 *
 * <p>However, interfaces without default methods are only initialized when their non-primitive
 * fields are accessed.
 *
 * <p>Test data for b/38255926
 */
public class DefaultInterfaceMethodWithStaticInitializer {

  final List<String> initializationOrder = new ArrayList<>();

  DefaultInterfaceMethodWithStaticInitializer register(Class<?> enclosingInterfaceClass) {
    initializationOrder.add(enclosingInterfaceClass.getSimpleName());
    return this;
  }

  private static long getTime() {
    return 0;
  }

  /** The simplest case: direct implementation. */
  public static class TestInterfaceSetOne {

    /**
     * A writable field so that other interfaces can set it in their static initializers.
     * (b/64290760)
     */
    static long writableStaticField;

    static final DefaultInterfaceMethodWithStaticInitializer RECORDER =
        new DefaultInterfaceMethodWithStaticInitializer();

    /** With a default method, this interface should run clinit. */
    interface I1 {
      long NOW = TestInterfaceSetOne.writableStaticField = getTime();
      DefaultInterfaceMethodWithStaticInitializer C = RECORDER.register(I1.class);

      default int defaultM1() {
        return 1;
      }
    }

    /** With a default method, this interface should run clinit. */
    interface I2 {
      long NOW = TestInterfaceSetOne.writableStaticField = getTime();
      DefaultInterfaceMethodWithStaticInitializer D = RECORDER.register(I2.class);

      default int defaultM2() {
        return 10;
      }
    }

    /** Class to trigger the clinit. */
    public static class C implements I1, I2 {
      public int sum() {
        return defaultM1() + defaultM2();
      }
    }

    public static ImmutableList<String> getExpectedInitializationOrder() {
      return ImmutableList.of(I1.class.getSimpleName(), I2.class.getSimpleName());
    }

    public static ImmutableList<String> getRealInitializationOrder() {
      return ImmutableList.copyOf(RECORDER.initializationOrder);
    }
  }

  /** Test for initializer execution order. */
  public static class TestInterfaceSetTwo {

    static final DefaultInterfaceMethodWithStaticInitializer RECORDER =
        new DefaultInterfaceMethodWithStaticInitializer();

    interface I1 {
      DefaultInterfaceMethodWithStaticInitializer C = RECORDER.register(I1.class);

      default int defaultM1() {
        return 1;
      }
    }

    interface I2 extends I1 {
      DefaultInterfaceMethodWithStaticInitializer D = RECORDER.register(I2.class);

      default int defaultM2() {
        return 2;
      }
    }

    /**
     * Loading this class will trigger the execution of the static initializers of I2 and I1.
     * However, I1 will be loaded first, as I2 extends I1.
     */
    public static class C implements I2, I1 {
      protected static final Integer INT_VALUE = Integer.valueOf(1); // To create a <clinit>

      public int sum() {
        return defaultM1() + defaultM2();
      }
    }

    public static ImmutableList<String> getExpectedInitializationOrder() {
      return ImmutableList.of(I1.class.getSimpleName(), I2.class.getSimpleName());
    }

    public static ImmutableList<String> getRealInitializationOrder() {
      return ImmutableList.copyOf(RECORDER.initializationOrder);
    }
  }

  /** Test: I2's <clinit> should not be executed. */
  public static class TestInterfaceSetThree {
    static final DefaultInterfaceMethodWithStaticInitializer RECORDER =
        new DefaultInterfaceMethodWithStaticInitializer();

    interface I1 {
      DefaultInterfaceMethodWithStaticInitializer C = RECORDER.register(I1.class);

      default int defaultM1() {
        return 6;
      }
    }

    interface I2 extends I1 {
      default int defaultM2() {
        return 5;
      }
    }

    /**
     * Loading this class will trigger the execution of the static initializers of I1. I2's will not
     * execute.
     */
    public static class C implements I2, I1 {
      protected static final Integer INT_VALUE = Integer.valueOf(1); // To create a <clinit>

      public int sum() {
        return defaultM1() + defaultM2();
      }
    }

    public static ImmutableList<String> getExpectedInitializationOrder() {
      return ImmutableList.of(I1.class.getSimpleName());
    }

    public static ImmutableList<String> getRealInitializationOrder() {
      return ImmutableList.copyOf(RECORDER.initializationOrder);
    }
  }
}
