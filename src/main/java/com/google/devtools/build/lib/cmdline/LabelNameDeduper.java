// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.cmdline;

import com.google.devtools.build.lib.runtime.MemoryOptimizations;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * Deduplicates target names in labels ("//pkg:name-to-dedup").
 *
 * <p>Uses a simple cache implemented via an array in a lock-free manner. Advantages: no measurable
 * CPU overhead, bounded memory overhead in the worst-case. Disadvantages: the deduping efficacy is
 * non-deterministic (depends on the order of [concurrent] calls).
 *
 * <p>This approach (and the specific cache size) were chosen experimentally, with many other
 * approaches considered and benchmarked. See b/545739944.
 */
public final class LabelNameDeduper {
  private static final int CACHE_SIZE = 262144;
  private static final AtomicReferenceArray<String> stringCache =
      new AtomicReferenceArray<>(CACHE_SIZE);

  private LabelNameDeduper() {}

  static String deduplicateTargetName(String name) {
    if (!MemoryOptimizations.allowNonDeterministicEfficacy.get()) {
      return name;
    }
    int hash = name.hashCode();
    int idx = (hash ^ (hash >>> 16)) & (CACHE_SIZE - 1);
    String existing = stringCache.getOpaque(idx);
    if (existing != null && existing.equals(name)) {
      return existing;
    }
    // Concurrent reads, writes, and clears are expected (for example, under memory pressure from
    // HighWaterMarkLimiter). Opaque access avoids a Java Memory Model data race without adding
    // memory barrier overhead, and String's final fields ensure safe publication.
    stringCache.setOpaque(idx, name);
    return name;
  }

  public static void clear() {
    for (int i = 0; i < CACHE_SIZE; i++) {
      stringCache.setOpaque(i, null);
    }
  }
}
