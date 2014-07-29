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
package com.google.devtools.build.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.Maps;

import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;

import javax.annotation.Nullable;

/**
 * An in-memory graph implementation. All operations are thread-safe with ConcurrentMap semantics.
 * Also see {@link ValueEntry}.
 * 
 * <p>This class is public only for use in alternative graph implementations.
 */
public class InMemoryGraph implements ProcessableGraph {

  protected final ConcurrentMap<SkyKey, ValueEntry> valueMap = Maps.newConcurrentMap();

  public InMemoryGraph() {
  }

  @Override
  public void remove(SkyKey skyKey) {
    valueMap.remove(skyKey);
  }

  @Override
  public ValueEntry get(SkyKey skyKey) {
    return valueMap.get(skyKey);
  }

  @Override
  public ValueEntry createIfAbsent(SkyKey key) {
    ValueEntry newval = new ValueEntry();
    ValueEntry oldval = valueMap.putIfAbsent(key, newval);
    return oldval == null ? newval : oldval;
  }

  /** Only done values exist to the outside world. */
  private static final Predicate<ValueEntry> NODE_DONE_PREDICATE =
      new Predicate<ValueEntry>() {
        @Override
        public boolean apply(ValueEntry entry) {
          return entry != null && entry.isDone();
        }
      };

  /**
   * Returns a value, if it exists. If not, returns null.
   */
  @Nullable public SkyValue getValue(SkyKey key) {
    ValueEntry entry = get(key);
    return NODE_DONE_PREDICATE.apply(entry) ? entry.getValue() : null;
  }

  /**
   * Returns a read-only live view of the values in the graph. All values are included. Dirty values
   * include their Value value. Values in error have a null value.
   */
  Map<SkyKey, SkyValue> getValues() {
    return Collections.unmodifiableMap(Maps.transformValues(
        valueMap,
        new Function<ValueEntry, SkyValue>() {
          @Override
          public SkyValue apply(ValueEntry entry) {
            return entry.toValueValue();
          }
        }));
  }

  /**
   * Returns a read-only live view of the done values in the graph. Dirty, changed, and error values
   * are not present in the returned map
   */
  Map<SkyKey, SkyValue> getDoneValues() {
    return Collections.unmodifiableMap(Maps.filterValues(Maps.transformValues(
        valueMap,
        new Function<ValueEntry, SkyValue>() {
          @Override
          public SkyValue apply(ValueEntry entry) {
            return entry.isDone() ? entry.getValue() : null;
          }
        }), Predicates.notNull()));
  }

  // Only for use by MemoizingEvaluator#delete
  Map<SkyKey, ValueEntry> getAllValues() {
    return Collections.unmodifiableMap(valueMap);
  }

  @VisibleForTesting
  protected ConcurrentMap<SkyKey, ValueEntry> getValueMap() {
    return valueMap;
  }
}
