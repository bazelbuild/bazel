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

public class InnerClassLambda {

  protected final List<String> reference;

  public InnerClassLambda(List<String> names) {
    this.reference = names;
  }

  /**
   * Uses a lambda that refers to a method parameter across 2 nested anonymous inner classes as well
   * as a field in the outer scope, the former being relatively unusual as it causes javac to emit 2
   * getfields to pass the captured parameter directly to the generated lambda class, covering an
   * unusual branch in how we rewrite invokedynamics.
   */
  public Function<List<String>, Callable<List<String>>> prefixFilter(String prefix) {
    return new Function<List<String>, Callable<List<String>>>() {
      @Override
      public Callable<List<String>> apply(List<String> input) {
        return new Callable<List<String>>() {
          @Override
          public List<String> call() throws Exception {
            return input.stream()
                .filter(n -> n.startsWith(prefix) && reference.contains(n))
                .collect(Collectors.toList());
          }
        };
      }
    };
  }
}
