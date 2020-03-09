// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar.testdata;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Lambda {

  private final List<String> names;

  public Lambda(List<String> names) {
    this.names = names;
  }

  public List<String> as() {
    return names.stream().filter(n -> n.startsWith("A")).collect(Collectors.toList());
  }

  public static Callable<String> hello() {
    return (Callable<String> & java.util.RandomAccess) () -> "hello";
  }

  public static Function<Integer, Callable<Long>> mult(int x) {
    return new Function<Integer, Callable<Long>>() {
      @Override
      public Callable<Long> apply(Integer y) {
        return () -> (long) x * (long) y;
      }
    };
  }

  /**
   * Test method for b/62456849. This method will first be converted to a synthetic method by {@link
   * com.google.devtools.build.android.desugar.Bug62456849TestDataGenerator}, and then Desugar
   * should keep it in this class without desugaring it (such as renaming).
   *
   * <p>Please ignore the lint error on the method name. The method name is intentionally chosen to
   * trigger a bug in Desugar.
   */
  public static int lambda$mult$0() {
    return 0;
  }
}
