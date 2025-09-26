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

import com.google.common.base.Preconditions;
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
public final class DeterministicHelper extends NotifyingHelper {
  public static final MemoizingEvaluator.GraphTransformerForTesting MAKE_DETERMINISTIC =
      makeTransformer(Listener.NULL_LISTENER, /*deterministic=*/ true);

  public static MemoizingEvaluator.GraphTransformerForTesting makeTransformer(
      Listener listener, boolean deterministic) {
    if (deterministic) {
      return new MemoizingEvaluator.GraphTransformerForTesting() {
        @Override
        public InMemoryGraph transform(InMemoryGraph graph) {
          return new DeterministicInMemoryGraph(graph, listener);
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

  private DeterministicHelper(Listener listener) {
    super(listener);
  }

  @Nullable
  @Override
  DeterministicNodeEntry wrapEntry(SkyKey key, @Nullable NodeEntry entry) {
    return entry == null ? null : new DeterministicNodeEntry(key, entry);
  }

  private static Map<SkyKey, ? extends NodeEntry> makeDeterministic(
      Map<SkyKey, ? extends NodeEntry> map) {
    Map<SkyKey, NodeEntry> result = new TreeMap<>(ALPHABETICAL_SKYKEY_COMPARATOR);
    result.putAll(map);
    Preconditions.checkState(
        map.size() == result.size(),
        "Different sky keys with identical toString results! Before=%s After=%s",
        result,
        map);
    return result;
  }

  static class DeterministicProcessableGraph extends NotifyingProcessableGraph {
    DeterministicProcessableGraph(ProcessableGraph delegate, Listener graphListener) {
      super(delegate, new DeterministicHelper(graphListener));
    }

    DeterministicProcessableGraph(ProcessableGraph delegate) {
      this(delegate, Listener.NULL_LISTENER);
    }

    @Override
    public NodeBatch getBatch(
        @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys)
        throws InterruptedException {
      NodeBatch batch = super.getBatch(requestor, reason, keys);
      var result = new TreeMap<SkyKey, NodeEntry>(ALPHABETICAL_SKYKEY_COMPARATOR);
      for (SkyKey key : keys) {
        NodeEntry entry = batch.get(key);
        if (entry != null) {
          result.put(key, entry);
        }
      }
      return result::get;
    }

    @Override
    public Map<SkyKey, ? extends NodeEntry> getBatchMap(
        @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys)
        throws InterruptedException {
      return makeDeterministic(super.getBatchMap(requestor, reason, keys));
    }
  }

  /**
   * This class uses TreeSet to store reverse dependencies of NodeEntry. As a result all values are
   * lexicographically sorted.
   */
  private class DeterministicNodeEntry extends NotifyingNodeEntry {
    private DeterministicNodeEntry(SkyKey myKey, NodeEntry delegate) {
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
        SkyValue value, Version graphVersion, @Nullable Version maxTransitiveSourceVersion)
        throws InterruptedException {
      TreeSet<SkyKey> result = new TreeSet<>(ALPHABETICAL_SKYKEY_COMPARATOR);
      result.addAll(super.setValue(value, graphVersion, maxTransitiveSourceVersion));
      return result;
    }

    @Override
    public NodeValueAndRdepsToSignal markClean() throws InterruptedException {
      TreeSet<SkyKey> result = new TreeSet<>(ALPHABETICAL_SKYKEY_COMPARATOR);
      NodeValueAndRdepsToSignal nodeValueAndRdepsToSignal = super.markClean();
      result.addAll(nodeValueAndRdepsToSignal.getRdepsToSignal());
      return new NodeValueAndRdepsToSignal(nodeValueAndRdepsToSignal.getValue(), result);
    }
  }
}
