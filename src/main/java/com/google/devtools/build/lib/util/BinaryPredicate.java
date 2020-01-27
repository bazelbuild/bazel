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
package com.google.devtools.build.lib.util;

import com.google.common.base.Predicate;

import javax.annotation.Nullable;

/**
 * A two-argument version of {@link Predicate} that determines a true or false value for pairs of
 * inputs.
 *
 * <p>Just as a {@link Predicate} is useful for filtering iterables of values, a {@link
 * BinaryPredicate} is useful for filtering iterables of paired values, like {@link
 * java.util.Map.Entry} or {@link Pair}.
 *
 * <p>See {@link Predicate} for implementation notes and advice.
 */
public interface BinaryPredicate<X, Y> {

  /**
   * Applies this {@link BinaryPredicate} to the given objects.
   *
   * @return the value of this predicate when applied to inputs {@code x, y}
   */
  boolean apply(@Nullable X x, @Nullable Y y);
}
