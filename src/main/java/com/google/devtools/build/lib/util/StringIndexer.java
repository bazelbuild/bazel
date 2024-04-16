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

import javax.annotation.Nullable;

/**
 * Provides bidirectional {@link String} ⇔ unique {@link Integer} mapping.
 *
 * <p>{@link Integer} is used in place of primitive {@code int} as it is assumed that indices will
 * be stored in maps. Passing {@link Integer} across API boundaries makes it less likely to store
 * duplicate instances and create garbage due to autoboxing.
 */
public interface StringIndexer {

  /** Removes all mappings. */
  void clear();

  /** Returns the number of strings in the index. */
  int size();

  /** Creates new mapping for the given string if necessary and returns string index. */
  Integer getOrCreateIndex(String s);

  /**
   * Returns the unique index for the given string if one was created via {@link #getOrCreateIndex},
   * or else {@code null}.
   */
  @Nullable
  Integer getIndex(String s);

  /**
   * Returns the string associated with the given index or {@code null} if it is not in the index.
   */
  @Nullable
  String getStringForIndex(Integer i);
}
