// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.xcode.util;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;

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
    return Iterables.concat(
        Iterables.transform(
            sequence,
            new Function<E, Iterable<E>>() {
              @Override
              public Iterable<E> apply(E element) {
                return ImmutableList.of(what, element);
              }
            }
        ));
  }
}
