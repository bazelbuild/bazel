// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.util.Preconditions;

/**
 * Utility code for interspersing items into sequences.
 */
public class Interspersing {
  private Interspersing() {}

  /**
   * Inserts {@code what} before each item in {@code sequence}, returning a lazy sequence of twice
   * the length.
   */
  public static <E> Iterable<E> beforeEach(final E what, Iterable<E> sequence) {
    Preconditions.checkNotNull(what);
    return Iterables.concat(
        Iterables.transform(sequence, element -> ImmutableList.of(what, element)));
  }

  /**
   * Prepends {@code what} to each string in {@code sequence}, returning a lazy sequence of the 
   * same length.
   */
  public static Iterable<String>
      prependEach(final String what, Iterable<String> sequence) {
    Preconditions.checkNotNull(what);
    return Iterables.transform(sequence, input -> what + input);
  }

  /**
   * Similar to {@link #prependEach(String, Iterable)}, but also converts each item in the sequence
   * to a string.
   */
  public static <E> Iterable<String>
      prependEach(String what, Iterable<E> sequence, Function<? super E, String> toString) {
    return prependEach(what, Iterables.transform(sequence, toString));
  }
}
