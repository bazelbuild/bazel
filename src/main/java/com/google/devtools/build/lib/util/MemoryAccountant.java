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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.util.ObjectGraphTraverser.EdgeType;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/** An object receiver for {@link ObjectGraphTraverser} that collects statistics about RAM use. */
public class MemoryAccountant implements ObjectGraphTraverser.ObjectReceiver {
  /** Measures the shallow size of an object. */
  public interface Measurer {

    /**
     * Return the shallow size of an objet.
     *
     * <p>If this instance doesn't know how to compute it, returns -1.
     */
    long maybeGetShallowSize(Object o);
  }

  /** Statistics collected about RAM use. */
  public static class Stats {
    private final Map<String, Long> objectCountByClass;
    private final Map<String, Long> memoryByClass;
    private long objectCount;
    private long memory;

    private Stats(boolean collectDetails) {
      objectCountByClass = collectDetails ? new HashMap<>() : null;
      memoryByClass = collectDetails ? new HashMap<>() : null;
      objectCount = 0L;
      memory = 0L;
    }

    private void addObject(String clazz, long size) {
      objectCount += 1;
      memory += size;
      if (objectCountByClass != null) {
        objectCountByClass.put(clazz, objectCountByClass.getOrDefault(clazz, 0L) + 1);
      }

      if (memoryByClass != null) {
        memoryByClass.put(clazz, memoryByClass.getOrDefault(clazz, 0L) + size);
      }
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

  private final ImmutableList<Measurer> measurers;
  private final Stats stats;

  public MemoryAccountant(Iterable<Measurer> measurers, boolean collectDetails) {
    this.measurers = ImmutableList.copyOf(measurers);
    stats = new Stats(collectDetails);
  }

  @Override
  public synchronized void objectFound(Object o, String context) {
    if (context == null) {
      if (o.getClass().isArray()) {
        context = "[] " + o.getClass().getComponentType().getName();
      } else {
        context = o.getClass().getName();
      }
    }

    long size = getShallowSize(o);
    stats.addObject(context, size);
  }

  private long getShallowSize(Object o) {
    for (Measurer measurer : measurers) {
      long candidate = measurer.maybeGetShallowSize(o);
      if (candidate >= 0) {
        return candidate;
      }
    }

    return ShallowObjectSizeComputer.getShallowSize(o);
  }

  @Override
  public void edgeFound(Object from, Object to, String toContext, EdgeType edgeType) {
    // Ignored for now.
  }

  public Stats getStats() {
    return stats;
  }
}
