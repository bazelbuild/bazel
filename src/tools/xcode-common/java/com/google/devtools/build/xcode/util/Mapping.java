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

package com.google.devtools.build.xcode.util;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;

import java.util.Map;

/**
 * Provides utility methods that make map lookup safe.
 */
public class Mapping {
  private Mapping() {
    throw new UnsupportedOperationException("static-only");
  }

  /**
   * Returns the value mapped to the given key for a map. If the mapping is not present, an absent
   * {@code Optional} is returned.
   * @throws NullPointerException if the map or key argument is null
   */
  public static <K, V> Optional<V> of(Map<K, V> map, K key) {
    Preconditions.checkNotNull(key);
    return Optional.fromNullable(map.get(key));
  }
}
