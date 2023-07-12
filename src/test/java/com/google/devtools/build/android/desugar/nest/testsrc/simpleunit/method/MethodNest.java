// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.nest.testsrc.simpleunit.method;

/** A nest for testing private method desugaring. */
@SuppressWarnings("MethodCanBeStatic") // Intentional for testing.
public class MethodNest {

  /** A nest member that encloses private methods with cross-mate invocations. */
  public static class MethodOwnerMate {

    static long staticMethod(long x, int y) {
      return x + y;
    }

    long instanceMethod(long x, int y) {
      return x + y;
    }

    private static long privateStaticMethod(long x, int y) {
      return x + y;
    }

    private long privateInstanceMethod(long x, int y) throws Exception {
      return x + y;
    }

    // No generation of bridge methods.
    private long inClassBoundInstanceMethod(long x) {
      return x == 0 ? 0 : x + inClassBoundInstanceMethod(x - 1);
    }

    // No generation of bridge methods.
    private static long inClassBoundStaticMethod(long x) {
      return x == 0 ? 0 : x + inClassBoundStaticMethod(x - 1);
    }
  }

  /** A nest member that has access to cross-mate private methods through inheritance. */
  public static class SubMate extends MethodOwnerMate {

    public static long invokePrivateStaticMethod(long x, int y) {
      return MethodOwnerMate.privateStaticMethod(x, y);
    }

    public long superAccessPrivateInstanceMethod(long x, int y) throws Exception {
      return 1 + super.privateInstanceMethod(x, y);
    }

    public long castAccessPrivateInstanceMethod(long x, int y) throws Exception {
      return 2 + ((MethodOwnerMate) this).privateInstanceMethod(x, y);
    }
  }

  public static long populatedFromInvokePrivateStaticMethod;
  public long populatedFromInvokePrivateInstanceMethod;

  // For testing a static initializer block.
  static {
    long t = MethodOwnerMate.privateStaticMethod(128L, 256);
    populatedFromInvokePrivateStaticMethod = 1 + t;
  }

  // For testing a instance initializer block.
  {
    try {
      populatedFromInvokePrivateInstanceMethod =
          new MethodOwnerMate().privateInstanceMethod(256L, 512);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @SuppressWarnings("StaticQualifiedUsingExpression") // Intentionally for testing.
  public static long invokeStaticMethod(long x, int y) {
    MethodOwnerMate methodOwnerMate = null;
    return methodOwnerMate.staticMethod(x, y);
  }

  public static long invokeInstanceMethod(MethodOwnerMate mateInstance, long x, int y) {
    return mateInstance.instanceMethod(x, y);
  }

  public static long invokePrivateStaticMethod(long x, int y) {
    return MethodOwnerMate.privateStaticMethod(x, y);
  }

  public static long invokePrivateInstanceMethod(MethodOwnerMate mateInstance, long x, int y)
      throws Exception {
    return mateInstance.privateInstanceMethod(x, y);
  }

  public static long invokeSuperAccessPrivateInstanceMethod(SubMate subMate, long x, int y)
      throws Exception {
    return subMate.superAccessPrivateInstanceMethod(x, y);
  }

  public static long invokeCastAccessPrivateInstanceMethod(SubMate subMate, long x, int y)
      throws Exception {
    return subMate.castAccessPrivateInstanceMethod(x, y);
  }

  public static void main(String[] args) {
    System.out.println("hello2");
  }
}
