// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect;

import com.google.common.collect.ForwardingMap;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A {@link Map} whose iteration order is the order in which entries were last inserted or updated.
 *
 * <p>This differs from {@link LinkedHashMap}, which keeps a re-inserted key at the position of its
 * original insertion. Use this when the last occurrence of a key (e.g. on the command line) should
 * also determine where that key is ordered relative to other keys.
 *
 * <p>The views returned by {@link #keySet}, {@link #values} and {@link #entrySet} are live views of
 * the underlying map. Updating a value through {@link Map.Entry#setValue} does not refresh the
 * entry's position.
 */
public final class RecencyMap<K, V> extends ForwardingMap<K, V> {
  private final Map<K, V> delegate = new LinkedHashMap<>();

  @Override
  protected Map<K, V> delegate() {
    return delegate;
  }

  @Override
  @Nullable
  public V put(K key, V value) {
    if (delegate.containsKey(key)) {
      V previous = delegate.remove(key);
      delegate.put(key, value);
      return previous;
    }
    return delegate.put(key, value);
  }

  @Override
  public void putAll(Map<? extends K, ? extends V> map) {
    standardPutAll(map);
  }
}
