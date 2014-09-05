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

import static com.google.common.base.Preconditions.checkNotNull;

import java.util.AbstractMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Map where only one value is allowed to be written per key, and any attempts to write a
 * conflicting value throw.
 *
 * Does not allow null keys or values.
 *
 * @param <K> the type of keys maintained by this map
 * @param <V> the type of mapped values
 */
public class WriteOneValueMap<K, V> extends AbstractMap<K, V> {

  private final ConcurrentHashMap<K, V> delegate;

  private WriteOneValueMap(ConcurrentHashMap<K, V> delegate) {
    this.delegate = checkNotNull(delegate);
  }

  public static <K, V> WriteOneValueMap<K, V> create() {
    return new WriteOneValueMap<K, V>(new ConcurrentHashMap<K, V>());
  }

  @Override
  public V get(Object key) {
    return delegate.get(key);
  }

  @Override
  public V put(K key, V value) {
    V oldValue = delegate.putIfAbsent(key, value);
    if (oldValue != null && !oldValue.equals(value)) {
      throw new IllegalArgumentException(String.format(
          "Tried to put <key=%s, value=%s> but already had value %s", key, value, oldValue));
    }
    return oldValue;
  }

  @Override
  public Set<Map.Entry<K, V>> entrySet() {
    return delegate.entrySet();
  }
}
