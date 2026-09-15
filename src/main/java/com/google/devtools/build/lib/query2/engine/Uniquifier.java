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
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

/** A helper for deduping values. */
@ThreadSafe
public interface Uniquifier<T> {
  /**
   * Returns whether {@code newElement} has been seen before by {@link #unique(T)} or
   * {@link #unique(Iterable)}.
   *
   * <p>Please note the difference between this method and {@link #unique(T)}!
   *
   * <p>This method is inherently racy wrt {@link #unique(T)} and {@link #unique(Iterable)}. Only
   * use it if you know what you are doing.
   */
  boolean uniquePure(T newElement);

  /**
   * Returns whether {@code newElement} has been seen before by {@link #unique(T)} or {@link
   * #unique(Iterable)}.
   */
  boolean unique(T newElement) throws QueryException;

  /**
   * Returns the subset of {@code newElements} that haven't been seen before by {@link #unique(T)}
   * or {@link #unique(Iterable)}.
   */
  ImmutableList<T> unique(Iterable<T> newElements) throws QueryException;
}
