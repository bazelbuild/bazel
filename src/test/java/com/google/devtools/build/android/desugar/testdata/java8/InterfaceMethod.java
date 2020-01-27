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
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Desugar test input interface that declares lambdas and method references in default and static
 * interface methods.
 */
public interface InterfaceMethod {
  public default List<String> defaultMethodReference(List<String> names) {
    return names.stream().filter(this::startsWithS).collect(Collectors.toList());
  }

  public default String defaultInvokingBootclasspathMethods(String expectedValue) {
    return Stream.of(expectedValue).findFirst().orElse("unexpected");
  }

  public default List<String> staticMethodReference(List<String> names) {
    return names.stream().filter(InterfaceMethod::startsWithA).collect(Collectors.toList());
  }

  public default List<String> lambdaCallsDefaultMethod(List<String> names) {
    return names.stream().filter(s -> startsWithS(s)).collect(Collectors.toList());
  }

  public static boolean startsWithA(String input) {
    return input.startsWith("A");
  }

  public default boolean startsWithS(String input) {
    return input.startsWith("S");
  }

  // Same descriptor as when method above is turned into static method
  public default boolean startsWithS(InterfaceMethod receiver, String input) {
    return startsWithS(input);
  }

  /**
   * Empty class implementing {@link InterfaceMethod} so the test can instantiate and call default
   * methods.
   */
  public static class Concrete implements InterfaceMethod {
    // We will rewrite this class to subclass InterfaceMethod's companion class, and call its super
    // constructor.  This field is here to make sure we don't also rewrite its constructor call.
    @SuppressWarnings("unused")
    private final Object o = new Object();
  }
}
