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

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * A map that orders entries by recency of insertion or update.
 */
public class RecencyMap<K, V> implements Map<K, V> {
  private final Map<K, V> backingMap = new LinkedHashMap<>();
  
  @Override
  public int size() {
    return backingMap.size();
  }

  @Override
  public boolean isEmpty() {
    return backingMap.isEmpty();
  }

  @Override
  public boolean containsKey(Object key) {
    return backingMap.containsKey(key);
  }

  @Override
  public boolean containsValue(Object value) {
    return backingMap.containsValue(value);
  }

  @Override
  public V get(Object key) {
    return backingMap.get(key);
  }

  @Override
  public V put(K k, V v) {
    if (backingMap.containsKey(k)) {
      backingMap.remove(k);
    }
    return backingMap.put(k, v);
  }

  @Override
  public V remove(Object key) {
    return backingMap.remove(key);
  }

  @Override
  public void putAll(Map<? extends K, ? extends V> m) {
    for (Map.Entry<? extends K, ? extends V> entry : m.entrySet()) {
      put(entry.getKey(), entry.getValue());
    }
  }

  @Override
  public void clear() {
    backingMap.clear();
  }

  @Override
  public Set<K> keySet() {
    return backingMap.keySet();
  }

  @Override
  public Collection<V> values() {
    return backingMap.values();
  }

  @Override
  public Set<Entry<K, V>> entrySet() {
    return backingMap.entrySet();
  }
}
