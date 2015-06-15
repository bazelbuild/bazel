// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.concurrent;

import com.google.common.base.Preconditions;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import javax.annotation.Nullable;

/**
 * An implementation of {@link KeyedFrequencyStore} that uses ref counting to efficiently only
 * store (key, value) pairs that have positive frequency.
 */
public class RefCountedHashMapKeyedFrequencyStore<K, V> implements KeyedFrequencyStore<K, V> {
  private final ConcurrentHashMap<K, ValueWithFrequency<V>> map = new ConcurrentHashMap<>();

  private static class ValueWithFrequency<V> {
    private final V value;
    private final AtomicInteger frequency;

    protected ValueWithFrequency(V value, int initialFrequency) {
      this.value = value;
      this.frequency = new AtomicInteger(initialFrequency);
    }
  }

  @Override
  public void put(K key, V value, int frequency) {
    Preconditions.checkState(frequency >= 0, frequency);
    if (frequency == 0) {
      map.remove(key);
    } else {
      map.put(key, new ValueWithFrequency<>(value, frequency));
    }
  }

  @Override
  @Nullable
  public V consume(K key) {
    ValueWithFrequency<V> vwf = map.get(key);
    if (vwf == null) {
      // Either the key isn't present or it has already been removed (because it has already
      // been consumed).
      return null;
    }
    int oldFrequency = vwf.frequency.getAndDecrement();
    if (oldFrequency <= 0) {
      // This can happen as a result of the following race: suppose the current frequency for key K
      // is F and T > F threads call consume(K) and all of them see the same object from the
      // map.get call above.. F-1 of these consume calls will decrement the frequency all the way
      // down to 1, one thread will "win the race" and decrement the frequency to 0 (see below
      // code), but the other T-F threads will be left with a "stale" ValueWithFrequency instance.
      // Since the value has already been exhaustively consumed, returning null is the appropriate
      // behavior here.
      return null;
    }
    if (oldFrequency == 1) {
      // We are the final consumer of the key.
      map.remove(key);
    }
    return vwf.value;
  }
}

