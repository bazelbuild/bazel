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

import com.google.devtools.build.android.desugar.testdata.separate.SeparateInterface;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ConcreteFunction implements SpecializedFunction<String, Long> {
  @Override
  public Long apply(String input) {
    return Long.valueOf(input);
  }

  // SpecializedParser makes it so we have to search multiple extended interfaces for bridge methods
  // when desugaring the lambda returned by this method.
  public static SpecializedParser<Integer> toInt() {
    return (s -> Integer.valueOf(s));
  }

  public static SeparateInterface<Long> isInt() {
    return (l -> Integer.MIN_VALUE <= l && l <= Integer.MAX_VALUE);
  }

  public static <T extends Number> List<T> parseAll(
      List<String> in, SpecializedFunction<String, T> parser) {
    return in.stream().map(parser).collect(Collectors.toList());
  }

  public static <T extends Number> List<T> doFilter(List<T> in, SeparateInterface<T> filter) {
    return in.stream().filter(filter).collect(Collectors.toList());
  }

  interface Parser<T> extends Function<String, T> {
    @Override
    public T apply(String in);
  }

  public interface SpecializedParser<T extends Number>
      extends SpecializedFunction<String, T>, Parser<T> {
    @Override
    public T apply(String in);
  }
}
