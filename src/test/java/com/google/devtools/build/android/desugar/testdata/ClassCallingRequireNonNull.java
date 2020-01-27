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

import java.util.Objects;
import java.util.function.IntSupplier;

/** This class is for the testing of desugaring calls to Objects.requireNonNull(Object o...) */
public class ClassCallingRequireNonNull {

  public static int getStringLengthWithMethodReference(String s) {
    return toInt(s::length);
  }

  public static int toInt(IntSupplier function) {
    return function.getAsInt();
  }

  public static int getStringLengthWithLambdaAndExplicitCallToRequireNonNull(final String s) {
    return toInt(() -> Objects.requireNonNull(s).length());
  }

  public static char getFirstCharVersionOne(String string) {
    Objects.requireNonNull(string);
    return string.charAt(0);
  }

  public static char getFirstCharVersionTwo(String string) {
    string = Objects.requireNonNull(string);
    return string.charAt(0);
  }

  public static char callRequireNonNullWithArgumentString(String string) {
    string = Objects.requireNonNull(string, "the string should not be null");
    return string.charAt(0);
  }

  public static char callRequireNonNullWithArgumentSupplier(String string) {
    string = Objects.requireNonNull(string, () -> "the string should not be null");
    return string.charAt(0);
  }
}
