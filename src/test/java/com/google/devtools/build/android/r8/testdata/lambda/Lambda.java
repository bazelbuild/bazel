// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8.testdata.lambda;

import java.util.function.Supplier;

/** Test class */
public final class Lambda {

  private Lambda() {}

  private static <T> T foo(Supplier<T> fn) {
    return fn.get();
  }

  public static void main(String[] args) {
    String unused = foo(() -> "Hello, world!");
  }
}
