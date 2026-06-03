// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;


import com.google.devtools.build.lib.concurrent.SettableFutureKeyedValue;
import java.util.concurrent.ConcurrentMap;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * A concurrent map of recursively populated futures or values.
 *
 * <p>This map achieves two goals in conjunction with the {@link SettableFutureKeyedValue}.
 *
 * <ul>
 *   <li>It ensures that for each key, the future is created and populated only once (assuming that
 *       the entry remains cached).
 *   <li>When the future succeeds, the future is unwrapped and only the value is retained.
 * </ul>
 *
 * <p>In the case of an error, the future is retained, according to the {@link #map} implementation.
 *
 * <p>{@code FutureT} must be a subclass of {@code ValueOrFutureT}. Unfortunately, this constraint
 * cannot be expressed using Java generics in conjunction with the constraint that {@code FutureT}
 * extends {@link SettableFutureKeyedValue}. It is the caller's responsibility to ensure this
 * relationship holds.
 */
abstract class AbstractValueOrFutureMap<
        KeyT,
        ValueOrFutureT,
        ValueT extends ValueOrFutureT,
        FutureT extends SettableFutureKeyedValue<FutureT, KeyT, ValueT>>
    implements BiConsumer<KeyT, ValueT> {

  private final ConcurrentMap<KeyT, ValueOrFutureT> map;
  private final ValueOrFutureFactory futureValueFactory;
  private final Class<FutureT> futureType;

  /**
   * Constructor.
   *
   * @param futureOrValueFactory creates appropriate instances of {@link SettableFutureKeyedValue}.
   *     The key and consumer parameters are provided for use in {@link SettableFutureKeyedValue}'s
   *     constructor.
   */
  AbstractValueOrFutureMap(
      ConcurrentMap<KeyT, ValueOrFutureT> map,
      BiFunction<KeyT, BiConsumer<KeyT, ValueT>, ValueOrFutureT> futureOrValueFactory,
      Class<FutureT> futureType) {
    this.map = map;
    this.futureValueFactory = new ValueOrFutureFactory(futureOrValueFactory);
    this.futureType = futureType;
  }

  Class<FutureT> futureType() {
    return futureType;
  }

  ValueOrFutureT getOrCreateValueForSubclasses(KeyT key) {
    ValueOrFutureT result = map.get(key);
    if (result == null) {
      ValueOrFutureT newValue = futureValueFactory.apply(key);
      result = map.putIfAbsent(key, newValue);
      if (result == null) {
        result = newValue;
      }
    }
    return result;
  }

  /**
   * @deprecated only used by {@link FutureValueFactory#apply}
   */
  @Override
  @Deprecated
  public void accept(KeyT key, ValueT value) {
    map.put(key, value);
  }

  private final class ValueOrFutureFactory implements Function<KeyT, ValueOrFutureT> {
    private final BiFunction<KeyT, BiConsumer<KeyT, ValueT>, ValueOrFutureT> futureOrValueFactory;

    private ValueOrFutureFactory(
        BiFunction<KeyT, BiConsumer<KeyT, ValueT>, ValueOrFutureT> futureOrValueFactory) {
      this.futureOrValueFactory = futureOrValueFactory;
    }

    @Override
    public ValueOrFutureT apply(KeyT key) {
      return futureOrValueFactory.apply(key, AbstractValueOrFutureMap.this);
    }
  }
}
