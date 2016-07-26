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
import java.util.EnumSet;
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
  static final MemoizingEvaluator.GraphTransformerForTesting MAKE_DETERMINISTIC =
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
        public InvalidatableGraph transform(InvalidatableGraph graph) {
          return new DeterministicInvalidatableGraph(graph, listener);
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

  private static final Comparator<SkyKey> ALPHABETICAL_SKYKEY_COMPARATOR =
      new Comparator<SkyKey>() {
        @Override
        public int compare(SkyKey o1, SkyKey o2) {
          return o1.toString().compareTo(o2.toString());
        }
      };

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

  private static Map<SkyKey, NodeEntry> makeDeterministic(Map<SkyKey, NodeEntry> map) {
    Map<SkyKey, NodeEntry> result = new TreeMap<>(ALPHABETICAL_SKYKEY_COMPARATOR);
    result.putAll(map);
    return result;
  }

  private static class DeterministicInvalidatableGraph extends NotifyingInvalidatableGraph {
    DeterministicInvalidatableGraph(InvalidatableGraph delegate, Listener graphListener) {
      super(delegate, new DeterministicHelper(graphListener));
    }

    @Override
    public Map<SkyKey, NodeEntry> getBatchForInvalidation(Iterable<SkyKey> keys) {
      return makeDeterministic(super.getBatchForInvalidation(keys));
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
    public Map<SkyKey, NodeEntry> createIfAbsentBatch(
        @Nullable SkyKey requestor, Reason reason, Iterable<SkyKey> keys) {
      return makeDeterministic(super.createIfAbsentBatch(requestor, reason, keys));
    }

    @Override
    public Map<SkyKey, NodeEntry> getBatchWithFieldHints(
        @Nullable SkyKey requestor,
        Reason reason,
        Iterable<SkyKey> keys,
        EnumSet<NodeEntryField> fields) {
      return makeDeterministic(super.getBatchWithFieldHints(requestor, reason, keys, fields));
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
