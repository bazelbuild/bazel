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

package com.google.devtools.build.android.desugar.nest.testsrc.simpleunit.interfacemethod;

import java.util.Collection;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

/** A nest for testing private interface methods desugaring. */
public class InterfaceNest {

  /** A class that implements the interface mate. */
  public static class ConcreteMate implements InterfaceMate {
    @Override
    public List<Long> mateValues() {
      return LongStream.range(0L, 100L).boxed().collect(Collectors.toList());
    }
  }
  /** A nest member that encloses private interface methods with cross-mate invocations. */
  interface SubInterfaceMateMate {
    private static long privateStaticMethod(long x, int y) {
      return x + y;
    }

    private long privateInstanceMethod(long x, int y) {
      return x + y;
    }
  }

  interface InterfaceMate {

    static long publicStaticMethod(long x, long y) {
      return x + y;
    }

    private static long privateStaticMethod(long x, int y) {
      return x + y;
    }

    private long privateInstanceMethod(long x, int y) {
      return x + y;
    }

    private static long privateStaticMethodWithLambdaEvaluation(
        Function<Long, Function<Long, Long>> hf, long a, long b) {
      return hf.apply(a).apply(b);
    }

    private static Function<Long, Function<Long, Long>> privateStaticMethodWithLambdaGeneration(
        long a, long b) {
      return p -> (q -> a + b * q);
    }

    private long privateInstanceMethodWithLambda(Collection<Long> factors, long a, long b) {
      Long sum = factors.stream().map(v -> a + v).reduce(0L, Long::sum);
      return mateValues().stream().mapToLong(v -> b + sum * v).sum();
    }

    List<Long> mateValues();

    private static long inClassBoundaryStaticMethod(long x) {
      return x == 0 ? 0 : x + inClassBoundaryStaticMethod(x - 1);
    }

    private long inClassBoundaryInstanceMethod(long x) {
      return x == 0 ? 0 : x + inClassBoundaryInstanceMethod(x - 1);
    }

    static long invokeStaticMethodInClassBoundary(long x) {
      return inClassBoundaryStaticMethod(x);
    }

    default long invokeInstanceMethodInClassBoundary(long x) {
      return inClassBoundaryInstanceMethod(x);
    }
  }

  public static long invokePublicStaticMethod(long x, int y) {
    return InterfaceMate.publicStaticMethod(x, y);
  }

  public static long invokePrivateStaticMethod(long x, int y) {
    return InterfaceMate.privateStaticMethod(x, y);
  }

  public static long invokePrivateInstanceMethod(InterfaceMate mate, long x, int y) {
    return mate.privateInstanceMethod(x, y);
  }

  public static long invokeSubMatePrivateStaticMethod(long x, int y) {
    return SubInterfaceMateMate.privateStaticMethod(x, y);
  }

  public static long invokeSubMatePrivateInstanceMethod(SubInterfaceMateMate mate, long x, int y) {
    return mate.privateInstanceMethod(x, y);
  }

  public static long invokePrivateStaticMethodWithLambda(long a0, long b0, long a1, long b1) {
    return InterfaceMate.privateStaticMethodWithLambdaEvaluation(
        InterfaceMate.privateStaticMethodWithLambdaGeneration(a0, b0), a1, b1);
  }

  public static long invokePrivateInstanceMethodWithLambda(
      InterfaceMate mate, Collection<Long> vals, long a, long b) {
    return mate.privateInstanceMethodWithLambda(vals, a, b);
  }

  public static long invokeInClassBoundaryStaticMethod(long x) {
    return InterfaceMate.invokeStaticMethodInClassBoundary(x);
  }

  public static long invokeInClassBoundaryInstanceMethod(InterfaceMate mate, long x) {
    return mate.invokeInstanceMethodInClassBoundary(x);
  }
}
