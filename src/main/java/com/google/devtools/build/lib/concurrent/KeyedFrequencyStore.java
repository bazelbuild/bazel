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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

import javax.annotation.Nullable;

/** A map from keys to values, with frequencies. */
public interface KeyedFrequencyStore<K, V> {
  @ConditionallyThreadSafe
  /**
   * Inserts {@code value} for the given {@code key} with the given non-negative {@code frequency}
   * (overwriting any existing value for that key).
   *
   * <p>Cannot be called concurrently with a call to {@code consume(key)}.
   */
  void put(K key, V value, int frequency);

  @Nullable
  @ThreadSafe
  /**
   * Removes for consumption one of the occurrences of the value for {@code key}, if any.
   *
   * <p>Formally, a call {@code consume(k)} returns {@code v} if it is the {@code f'}th such call
   * since a call to {@code put(k, v, f)} (with {@code f' <= f}), and {@code null} otherwise.
   */
  V consume(K key);
}