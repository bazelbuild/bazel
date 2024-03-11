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

package com.google.devtools.build.lib.util;

import com.google.devtools.build.lib.util.ObjectGraphTraverser.EdgeType;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/** An object receiver for {@link ObjectGraphTraverser} that collects statistics about RAM use. */
public class MemoryAccountant implements ObjectGraphTraverser.ObjectReceiver {

  /** Statistics collected about RAM use. */
  public static class Stats {
    private final Map<String, Long> objectCountByClass;
    private final Map<String, Long> memoryByClass;
    private long objectCount;
    private long memory;

    private Stats() {
      objectCountByClass = new HashMap<>();
      memoryByClass = new HashMap<>();
      objectCount = 0L;
      memory = 0L;
    }

    private void addObject(String clazz, long size) {
      objectCount += 1;
      memory += size;
      objectCountByClass.put(clazz, objectCountByClass.getOrDefault(clazz, 0L) + 1);
      memoryByClass.put(clazz, memoryByClass.getOrDefault(clazz, 0L) + size);
    }

    public long getObjectCount() {
      return objectCount;
    }

    public long getMemoryUse() {
      return memory;
    }

    public Map<String, Long> getObjectCountByClass() {
      return Collections.unmodifiableMap(objectCountByClass);
    }

    public Map<String, Long> getMemoryByClass() {
      return Collections.unmodifiableMap(memoryByClass);
    }
  }

  private final Stats stats;

  public MemoryAccountant() {
    stats = new Stats();
  }

  @Override
  public void objectFound(Object o, String context) {
    if (context == null) {
      context = o.getClass().getName();
    }

    long size = ShallowObjectSizeComputer.getShallowSize(o);
    stats.addObject(context, size);
  }

  @Override
  public void edgeFound(Object from, Object to, String toContext, EdgeType edgeType) {
    // Ignored for now.
  }

  public Stats getStats() {
    return stats;
  }
}
