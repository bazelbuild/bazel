// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Function;
import com.google.common.collect.Iterables;

import java.util.Collection;
import java.util.Comparator;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import javax.annotation.Nullable;

/**
 * {@link NotifyingGraph} that returns reverse deps, temporary direct deps, and the results of
 * batch requests ordered alphabetically by sky key string representation.
 */
public class DeterministicGraph<TGraph extends ThinNodeQueryableGraph>
    extends NotifyingGraph<TGraph> {
  public static final Function<ThinNodeQueryableGraph, ProcessableGraph> MAKE_DETERMINISTIC =
      new Function<ThinNodeQueryableGraph, ProcessableGraph>() {
        @Override
        public ProcessableGraph apply(ThinNodeQueryableGraph queryableGraph) {
          if (queryableGraph instanceof InMemoryGraph) {
            return new DeterministicInMemoryGraph((InMemoryGraph) queryableGraph);
          } else {
            return new DeterministicGraph<>(queryableGraph);
          }
        }
      };

  public static Function<ThinNodeQueryableGraph, ProcessableGraph> makeTransformer(
      final Listener listener, boolean deterministic) {
    if (deterministic) {
      return new Function<ThinNodeQueryableGraph, ProcessableGraph>() {
        @Override
        public ProcessableGraph apply(ThinNodeQueryableGraph queryableGraph) {
          if (queryableGraph instanceof InMemoryGraph) {
            return new DeterministicInMemoryGraph((InMemoryGraph) queryableGraph, listener);
          } else {
            return new DeterministicGraph<>(queryableGraph, listener);
          }
        }
      };
    } else {
      return NotifyingGraph.makeNotifyingTransformer(listener);
    }
  }

  private static final Comparator<SkyKey> ALPHABETICAL_SKYKEY_COMPARATOR =
      new Comparator<SkyKey>() {
        @Override
        public int compare(SkyKey o1, SkyKey o2) {
          return o1.toString().compareTo(o2.toString());
        }
      };

  DeterministicGraph(TGraph delegate, Listener listener) {
    super(delegate, listener);
  }

  DeterministicGraph(TGraph delegate) {
    super(delegate, NotifyingGraph.Listener.NULL_LISTENER);
  }

  @Nullable
  @Override
  protected DeterministicValueEntry wrapEntry(SkyKey key, @Nullable ThinNodeEntry entry) {
    return entry == null ? null : new DeterministicValueEntry(key, entry);
  }

  private static Map<SkyKey, NodeEntry> makeDeterministic(Map<SkyKey, NodeEntry> map) {
    Map<SkyKey, NodeEntry> result = new TreeMap<>(ALPHABETICAL_SKYKEY_COMPARATOR);
    result.putAll(map);
    return result;
  }

  @Override
  public Map<SkyKey, NodeEntry> getBatch(Iterable<SkyKey> keys) {
    return makeDeterministic(super.getBatch(keys));
  }

  @Override
  public Map<SkyKey, NodeEntry> createIfAbsentBatch(Iterable<SkyKey> keys) {
    return makeDeterministic(super.createIfAbsentBatch(keys));
  }

  /**
   * This class uses TreeSet to store reverse dependencies of NodeEntry. As a result all values are
   * lexicographically sorted.
   */
  private class DeterministicValueEntry extends NotifyingNodeEntry {
    private DeterministicValueEntry(SkyKey myKey, ThinNodeEntry delegate) {
      super(myKey, delegate);
    }

    @Override
    public synchronized Collection<SkyKey> getReverseDeps() {
      TreeSet<SkyKey> result = new TreeSet<>(ALPHABETICAL_SKYKEY_COMPARATOR);
      Iterables.addAll(result, super.getReverseDeps());
      return result;
    }

    @Override
    public synchronized Set<SkyKey> getInProgressReverseDeps() {
      TreeSet<SkyKey> result = new TreeSet<>(ALPHABETICAL_SKYKEY_COMPARATOR);
      result.addAll(super.getInProgressReverseDeps());
      return result;
    }
  }
}
