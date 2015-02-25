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
public class DeterministicInMemoryGraph extends NotifyingInMemoryGraph {
  public DeterministicInMemoryGraph(Listener listener) {
    super(listener);
  }

  public DeterministicInMemoryGraph() {
    super(Listener.NULL_LISTENER);
  }

  @Override
  protected DeterministicValueEntry getEntry(SkyKey key) {
    return new DeterministicValueEntry(key);
  }

  /**
   * This class uses TreeSet to store reverse dependencies of NodeEntry. As a result all values are
   * lexicographically sorted.
   */
  private class DeterministicValueEntry extends NotifyingNodeEntry {
    private DeterministicValueEntry(SkyKey myKey) {
      super(myKey);
    }

    final Comparator<SkyKey> valueEntryComparator = new Comparator<SkyKey>() {
      @Override
      public int compare(SkyKey o1, SkyKey o2) {
        return o1.toString().compareTo(o2.toString());
      }
    };
    @SuppressWarnings("unchecked")
    @Override
    synchronized Iterable<SkyKey> getReverseDeps() {
      TreeSet<SkyKey> result = new TreeSet<SkyKey>(valueEntryComparator);
      if (reverseDeps instanceof List) {
        result.addAll((Collection<? extends SkyKey>) reverseDeps);
      } else {
        result.add((SkyKey) reverseDeps);
      }
      return result;
    }

    @Override
    synchronized Set<SkyKey> getInProgressReverseDeps() {
      TreeSet<SkyKey> result = new TreeSet<SkyKey>(valueEntryComparator);
      result.addAll(buildingState.getReverseDepsToSignal());
      return result;
    }
  }
}
