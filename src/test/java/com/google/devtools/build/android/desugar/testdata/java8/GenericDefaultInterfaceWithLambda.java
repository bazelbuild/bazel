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

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/** An interface with default methods, lambdas, and generics */
public interface GenericDefaultInterfaceWithLambda<T> {

  T getBaseValue();

  T increment(T value);

  String toString(T value);

  public default ArrayList<T> toList(int bound) {
    ArrayList<T> result = new ArrayList<>();
    if (bound == 0) {
      return result;
    }
    result.add(getBaseValue());
    for (int i = 1; i < bound; ++i) {
      result.add(increment(result.get(i - 1)));
    }
    return result;
  }

  public default List<String> convertToStringList(List<T> list) {
    return list.stream().map(this::toString).collect(Collectors.toList());
  }

  public default Function<Integer, ArrayList<T>> toListSupplier() {
    return this::toList;
  }

  /** The type parameter is concretized to {@link Number} */
  interface LevelOne<T extends Number> extends GenericDefaultInterfaceWithLambda<T> {}

  /** The type parameter is instantiated to {@link Integer} */
  interface LevelTwo extends LevelOne<Integer> {

    @Override
    default Integer getBaseValue() {
      return 0;
    }
  }

  /** An abstract class with no implementing methods. */
  abstract static class ClassOne implements LevelTwo {}

  /** A class for {@link Integer} */
  class ClassTwo extends ClassOne {

    @Override
    public Integer increment(Integer value) {
      return value + 1;
    }

    @Override
    public String toString(Integer value) {
      return value.toString();
    }
  }

  /** A class fo {@link Long} */
  class ClassThree implements LevelOne<Long> {

    @Override
    public Long getBaseValue() {
      return Long.valueOf(0);
    }

    @Override
    public Long increment(Long value) {
      return value + 1;
    }

    @Override
    public String toString(Long value) {
      return value.toString();
    }
  }
}
