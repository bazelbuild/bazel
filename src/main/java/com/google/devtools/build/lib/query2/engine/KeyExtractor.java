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
package com.google.devtools.build.lib.query2.engine;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

/**
 * Helper for extracting a key of type {@code K} from an element of type {@code T}.
 *
 * <p>Depending on the choice of {@code K}, this enables potential memory optimizations.
 */
@ThreadSafe
public interface KeyExtractor<T, K> {
  /** Extracts an unique key that can be used to dedupe the given {@code element}. */
  K extractKey(T element);

  static <T1, T2, K> KeyExtractor<T1, K> compose(
      KeyExtractor<T1, ? extends T2> inner, KeyExtractor<T2, K> outer) {
    return t1 -> outer.extractKey(inner.extractKey(t1));
  }
}
