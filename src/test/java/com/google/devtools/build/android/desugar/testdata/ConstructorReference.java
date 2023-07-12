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

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class ConstructorReference {

  private final List<String> names;

  private ConstructorReference(String name) {
    names = new ArrayList<>(1);
    names.add(name);
  }

  public ConstructorReference(List<String> names) {
    this.names = names;
  }

  public List<Integer> toInt() {
    return names.stream().map(Integer::new).collect(Collectors.toList());
  }

  public static Function<String, ConstructorReference> singleton() {
    return ConstructorReference::new;
  }

  public static Supplier<ConstructorReference> emptyThroughJavacGeneratedBridge() {
    // Because Empty is private in another (inner) class, Javac seems to generate a lambda body
    // method in this case that calls the Empty(SentinalType) bridge constructor Javac generates.
    return Empty::new;
  }

  private static class Empty extends ConstructorReference {

    private Empty() {
      super(new ArrayList<String>(0));
      throw new RuntimeException("got it!");
    }
  }
}
