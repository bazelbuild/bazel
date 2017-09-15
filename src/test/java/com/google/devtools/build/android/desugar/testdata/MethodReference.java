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

import com.google.devtools.build.android.desugar.testdata.separate.SeparateBaseClass;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class MethodReference extends SeparateBaseClass<String> {

  private final List<String> names;

  public MethodReference(List<String> names) {
    super(names);
    this.names = names;
  }

  // Class method reference
  public void appendAll(StringBuilder dest) {
    names.stream().forEach(dest::append);
  }

  // Interface method reference (regression test for b/33304582)
  public List<String> transform(Transformer<String> transformer) {
    return names.stream().map(transformer::transform).collect(Collectors.toList());
  }

  // Private method reference (regression test for b/33378312)
  public List<String> some() {
    return names.stream().filter(MethodReference::startsWithS).collect(Collectors.toList());
  }

  // Protected method reference in a base class of another package (regression test for b/33378312)
  public List<String> intersect(List<String> other) {
    return other.stream().filter(this::contains).collect(Collectors.toList());
  }

  // Contains the same method reference as intersect
  public List<String> onlyIn(List<String> other) {
    Predicate<String> p = this::contains;
    return other.stream().filter(p.negate()).collect(Collectors.toList());
  }

  // Private method reference to an instance method that throws (regression test for b/33378312)
  public Callable<String> stringer() {
    return this::throwing;
  }

  /** Returns a method reference derived from an expression (object.toString()). */
  public static Function<Integer, Character> stringChars(Object object) {
    return (object == null ? "" : object.toString())::charAt;
  }

  /** Returns a method reference derived from a field */
  public Predicate<String> toPredicate() {
    return names::contains;
  }

  private static boolean startsWithS(String input) {
    return input.startsWith("S");
  }

  private String throwing() throws Exception {
    StringBuilder msg = new StringBuilder();
    appendAll(msg);
    throw new IOException(msg.toString());
  }

  /** Interface to create a method reference for in {@link #transform}. */
  public interface Transformer<T> {
    T transform(T input);
  }
}
