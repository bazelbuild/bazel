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

package com.google.devtools.build.lib.collect;

import java.util.Iterator;

/**
 * A minimal map interface that avoids methods whose implementation tends to force GC churn, or
 * otherwise overly constrain implementation freedom.
 *
 * <p>TODO: Convert to interface once we move to Java 8.
 */
public abstract class CompactImmutableMap<K, V> implements Iterable<K> {

  public boolean containsKey(K key) {
    return get(key) != null;
  }

  public abstract V get(K key);

  public abstract int size();

  public abstract K keyAt(int index);

  public abstract V valueAt(int index);

  @Override
  public Iterator<K> iterator() {
    return new ImmutableMapSharedKeysIterator();
  }

  private class ImmutableMapSharedKeysIterator implements Iterator<K> {
    int index = 0;

    @Override
    public boolean hasNext() {
      return index < size();
    }

    @Override
    public K next() {
      K key = keyAt(index);
      ++index;
      return key;
    }
  }
}
