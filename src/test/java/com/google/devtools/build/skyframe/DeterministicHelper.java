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

import com.google.common.collect.Iterables;
import java.util.Collection;
import java.util.Comparator;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import javax.annotation.Nullable;

/**
 * {@link NotifyingHelper} that returns reverse deps, temporary direct deps, and the results of
 * batch requests ordered alphabetically by sky key string representation.
 */
public class DeterministicHelper extends NotifyingHelper {
  public static final MemoizingEvaluator.GraphTransformerForTesting MAKE_DETERMINISTIC =
      makeTransformer(Listener.NULL_LISTENER, /*deterministic=*/ true);

  public static MemoizingEvaluator.GraphTransformerForTesting makeTransformer(
      final Listener listener, boolean deterministic) {
    if (deterministic) {
      return new MemoizingEvaluator.GraphTransformerForTesting() {
        @Override
        public InMemoryGraph transform(InMemoryGraph graph) {
          return new DeterministicInMemoryGraph(graph, listener);
        }

        @Override
        public QueryableGraph transform(QueryableGraph graph) {
          return new DeterministicQueryableGraph(graph, listener);
        }

        @Override
        public ProcessableGraph transform(ProcessableGraph graph) {
          return new DeterministicProcessableGraph(graph, listener);
        }
      };
    } else {
      return NotifyingHelper.makeNotifyingTransformer(listener);
    }
  }

  /** Compare using SkyKey argument first, so that tests can easily order keys. */
  private static final Comparator<SkyKey> ALPHABETICAL_SKYKEY_COMPARATOR =
      Comparator.<SkyKey, String>comparing(key -> key.argument().toString())
          .thenComparing(key -> key.functionName().toString());

  DeterministicHelper(Listener listener) {
    super(listener);
  }

  DeterministicHelper() {
    super(NotifyingHelper.Listener.NULL_LISTENER);
  }

  @Nullable
  @Override
  protected DeterministicValueEntry wrapEntry(SkyKey key, @Nullable ThinNodeEntry entry) {
    return entry == null ? null : new DeterministicValueEntry(key, entry);
  }

  private static Map<SkyKey, ? extends NodeEntry> makeDeterministic(
      Map<SkyKey, ? extends NodeEntry> map) {
    Map<SkyKey, NodeEntry> result = new TreeMap<>(ALPHABETICAL_SKYKEY_COMPARATOR);
    result.putAll(map);
    return result;
  }

  private static class DeterministicQueryableGraph extends NotifyingQueryableGraph {
    DeterministicQueryableGraph(QueryableGraph delegate, Listener graphListener) {
      super(delegate, new DeterministicHelper(graphListener));
    }

    @Override
    public Map<SkyKey, ? extends NodeEntry> getBatch(
        @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys)
            throws InterruptedException {
      return makeDeterministic(super.getBatch(requestor, reason, keys));
    }
  }

  static class DeterministicProcessableGraph extends NotifyingProcessableGraph {
    DeterministicProcessableGraph(ProcessableGraph delegate, Listener graphListener) {
      super(delegate, new DeterministicHelper(graphListener));
    }

    DeterministicProcessableGraph(ProcessableGraph delegate) {
      this(delegate, Listener.NULL_LISTENER);
    }

    @Override
    public void remove(SkyKey key) {
      delegate.remove(key);
    }

    @Override
    public Map<SkyKey, ? extends NodeEntry> createIfAbsentBatch(
        @Nullable SkyKey requestor, Reason reason, Iterable<SkyKey> keys)
        throws InterruptedException {
      return makeDeterministic(super.createIfAbsentBatch(requestor, reason, keys));
    }

    @Override
    public Map<SkyKey, ? extends NodeEntry> getBatch(
        @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys)
            throws InterruptedException {
      return makeDeterministic(super.getBatch(requestor, reason, keys));
    }
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
    public synchronized Collection<SkyKey> getReverseDepsForDoneEntry()
        throws InterruptedException {
      TreeSet<SkyKey> result = new TreeSet<>(ALPHABETICAL_SKYKEY_COMPARATOR);
      Iterables.addAll(result, super.getReverseDepsForDoneEntry());
      return result;
    }

    @Override
    public synchronized Set<SkyKey> getInProgressReverseDeps() {
      TreeSet<SkyKey> result = new TreeSet<>(ALPHABETICAL_SKYKEY_COMPARATOR);
      result.addAll(super.getInProgressReverseDeps());
      return result;
    }

    @Override
    public Set<SkyKey> setValue(
        SkyValue value, Version version, DepFingerprintList depFingerprintList)
        throws InterruptedException {
      TreeSet<SkyKey> result = new TreeSet<>(ALPHABETICAL_SKYKEY_COMPARATOR);
      result.addAll(super.setValue(value, version, depFingerprintList));
      return result;
    }
  }
}
