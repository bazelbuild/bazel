// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.engine;

import com.google.common.collect.ImmutableList;

/**
 * A class used for deduplication of {@code Iterable}s. If called with repeated elements or multiple
 * times with repeated elements between calls, it guarantees to output only those elements exactly once.
 */
public interface Uniquifier<T> {

  /**
   * Receives an iterable and returns the list of elements that were not already seen. The
   * uniqueness need to be guaranteed for elements of the same iterable and multiple calls to the
   * {@code unique} method.
   *
   * @param newElements The new elements to process.
   * @return The subset of elements not already seen by this Uniquifier.
   */
  ImmutableList<T> unique(Iterable<T> newElements);
}
