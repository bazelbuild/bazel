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

import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/** {@link NotifyingInMemoryGraph} that returns reverse deps ordered alphabetically. */
class DeterministicInMemoryGraph extends NotifyingInMemoryGraph {
  public DeterministicInMemoryGraph(Listener listener) {
    super(listener);
  }

  public DeterministicInMemoryGraph() {
    super(Listener.NULL_LISTENER);
  }

  @Override
  protected DeterministicNodeEntry getEntry(NodeKey key) {
    return new DeterministicNodeEntry(key);
  }

  /**
   * This class uses TreeSet to store reverse dependencies of NodeEntry. As a result all nodes are
   * lexicographically sorted.
   */
  private class DeterministicNodeEntry extends NotifyingNodeEntry {
    private DeterministicNodeEntry(NodeKey myKey) {
      super(myKey);
    }

    final Comparator<NodeKey> nodeEntryComparator = new Comparator<NodeKey>() {
      @Override
      public int compare(NodeKey o1, NodeKey o2) {
        return o1.toString().compareTo(o2.toString());
      }
    };
    @SuppressWarnings("unchecked")
    @Override
    synchronized Iterable<NodeKey> getReverseDeps() {
      TreeSet<NodeKey> result = new TreeSet<NodeKey>(nodeEntryComparator);
      if (reverseDeps instanceof List) {
        result.addAll((Collection<? extends NodeKey>) reverseDeps);
      } else {
        result.add((NodeKey) reverseDeps);
      }
      return result;
    }

    @Override
    synchronized Set<NodeKey> getInProgressReverseDeps() {
      TreeSet<NodeKey> result = new TreeSet<NodeKey>(nodeEntryComparator);
      result.addAll(buildingState.getReverseDepsToSignal());
      return result;
    }
  }
}
