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

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/** Desugaring test input interface that includes a default method and a static method. */
public interface FunctionWithDefaultMethod<T extends Number> extends Function<T, T> {

  @Override
  T apply(T input);

  static <T extends Number> Function<T, Long> toLong() {
    return input -> input.longValue();
  }

  default T twice(T input) {
    return apply(apply(input));
  }

  /** Don't call this method from tests, it won't work since Desugar moves it! */
  static FunctionWithDefaultMethod<Integer> inc(int add) {
    return input -> input + add;
  }

  /**
   * Implementation of {@link FunctionWithDefaultMethod} that overrides the default method. Also
   * declares static methods the test uses to exercise the code in this file.
   */
  public static class DoubleInts implements FunctionWithDefaultMethod<Integer> {
    @Override
    public Integer apply(Integer input) {
      return 2 * input;
    }

    @Override
    public Integer twice(Integer input) {
      return 5 * input; // deliberately wrong :)
    }

    public static List<Long> add(List<Integer> ints, int add) {
      return ints.stream().map(inc(add)).map(toLong()).collect(Collectors.toList());
    }

    public static FunctionWithDefaultMethod<Integer> doubleLambda() {
      return input -> 2 * input;
    }

    public static FunctionWithDefaultMethod<Integer> incTwice(int add) {
      return inc(add)::twice;
    }

    public static FunctionWithDefaultMethod<Integer> times5() {
      return new DoubleInts2()::twice;
    }

    public static Function<Integer, FunctionWithDefaultMethod<Integer>> incFactory() {
      return FunctionWithDefaultMethod::inc;
    }
  }

  /** Empty subclass that explicitly implements the interface the superclass already implements. */
  public static class DoubleInts2 extends DoubleInts
      implements FunctionWithDefaultMethod<Integer> {}
}
