// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.Label.LabelInterner;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.PooledInterner;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.supplier.InterruptibleSupplier;
import com.google.devtools.build.skyframe.SkyKey.SkyKeyInterner;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.ForOverride;
import java.time.Duration;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * An in-memory graph implementation. All operations are thread-safe with ConcurrentMap semantics.
 * Also see {@link NodeEntry}.
 *
 * <p>This class is public only for use in alternative graph implementations.
 */
public class InMemoryGraphImpl implements InMemoryGraph {

  private static final int PARALLELISM_THRESHOLD = 1024;

  protected final ConcurrentHashMap<SkyKey, InMemoryNodeEntry> nodeMap;
  private final NodeBatch getBatch;
  private final NodeBatch createIfAbsentBatch;
  private final boolean usePooledInterning;

  InMemoryGraphImpl() {
    this(/* initialCapacity= */ 1 << 10);
  }

  /**
   * For some shell integration tests, we don't want to apply {@link SkyKeyInterner} created and
   * bind {@code SkyKeyInterner#globalPool} to the second {@link InMemoryGraph}.
   */
  InMemoryGraphImpl(boolean usePooledInterning) {
    this(/* initialCapacity= */ 1 << 10, usePooledInterning);
  }

  protected InMemoryGraphImpl(int initialCapacity) {
    this(initialCapacity, /* usePooledInterning= */ true);
  }

  private InMemoryGraphImpl(int initialCapacity, boolean usePooledInterning) {
    this.nodeMap = new ConcurrentHashMap<>(initialCapacity);
    this.getBatch = nodeMap::get;
    this.createIfAbsentBatch = this::createIfAbsent;
    this.usePooledInterning = usePooledInterning;
    if (usePooledInterning) {
      SkyKeyInterner.setGlobalPool(new SkyKeyPool());
      LabelInterner.setGlobalPool(new LabelPool());
    }
  }

  @Override
  public void remove(SkyKey skyKey) {
    weakInternSkyKey(skyKey);
    InMemoryNodeEntry nodeEntry = nodeMap.remove(skyKey);
    if (skyKey instanceof PackageIdentifier && nodeEntry != null) {
      weakInternPackageTargetsLabels(
          (PackageValue) nodeEntry.toValue()); // Dirty or changed value are needed.
    }
  }

  @Override
  public void removeIfDone(SkyKey key) {
    nodeMap.computeIfPresent(
        key,
        (k, e) -> {
          if (e.isDone()) {
            weakInternSkyKey(k);
            if (k instanceof PackageIdentifier) {
              weakInternPackageTargetsLabels((PackageValue) e.toValue());
            }
            return null;
          }
          return e;
        });
  }

  private void weakInternSkyKey(SkyKey skyKey) {
    if (!usePooledInterning) {
      return;
    }
    SkyKeyInterner<?> interner = skyKey.getSkyKeyInterner();
    if (interner != null) {
      interner.weakInternUnchecked(skyKey);
    }
  }

  private void weakInternPackageTargetsLabels(@Nullable PackageValue packageValue) {
    if (!usePooledInterning || packageValue == null) {
      return;
    }
    LabelInterner interner = Label.getLabelInterner();

    ImmutableSortedMap<String, Target> targets = packageValue.getPackage().getTargets();
    targets.values().forEach(t -> interner.weakIntern(t.getLabel()));
  }

  @Override
  public NodeEntry get(@Nullable SkyKey requestor, Reason reason, SkyKey skyKey) {
    return nodeMap.get(skyKey);
  }

  @Override
  public NodeBatch getBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys) {
    return getBatch;
  }

  @Override
  public InterruptibleSupplier<NodeBatch> getBatchAsync(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys) {
    return () -> getBatch;
  }

  @Override
  public Map<SkyKey, NodeEntry> getBatchMap(
      SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys) {
    // Use a HashMap, not an ImmutableMap.Builder, because we have not yet deduplicated these keys
    // and ImmutableMap.Builder does not tolerate duplicates. The map will be thrown away shortly.
    HashMap<SkyKey, NodeEntry> result = new HashMap<>();
    for (SkyKey key : keys) {
      NodeEntry entry = get(null, Reason.OTHER, key);
      if (entry != null) {
        result.put(key, entry);
      }
    }
    return result;
  }

  @ForOverride
  protected InMemoryNodeEntry newNodeEntry(SkyKey key) {
    return new IncrementalInMemoryNodeEntry(key);
  }

  @Override
  @CanIgnoreReturnValue
  public NodeBatch createIfAbsentBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys) {
    for (SkyKey key : keys) {
      createIfAbsent(key);
    }
    return createIfAbsentBatch;
  }

  @CanIgnoreReturnValue
  private InMemoryNodeEntry createIfAbsent(SkyKey skyKey) {
    SkyKeyInterner<?> interner = skyKey.getSkyKeyInterner();
    if (!usePooledInterning || interner == null) {
      return nodeMap.computeIfAbsent(skyKey, this::newNodeEntry);
    }

    // The key is typically already present. Record whether this thread newly created a node so that
    // we can skip calling removeWeak if it was already present.
    boolean[] newlyCreated = new boolean[1];
    InMemoryNodeEntry nodeEntry =
        nodeMap.computeIfAbsent(
            skyKey,
            k -> {
              newlyCreated[0] = true;
              return newNodeEntry(k);
            });
    if (newlyCreated[0]) {
      interner.removeWeak(skyKey);
    }
    return nodeEntry;
  }

  @Override
  public DepsReport analyzeDepsDoneness(SkyKey parent, Collection<SkyKey> deps) {
    return DepsReport.NO_INFORMATION;
  }

  @Override
  public int valuesSize() {
    return nodeMap.size();
  }

  @Override
  public Map<SkyKey, SkyValue> getValues() {
    return Collections.unmodifiableMap(Maps.transformValues(nodeMap, InMemoryNodeEntry::toValue));
  }

  @Override
  public Map<SkyKey, SkyValue> getDoneValues() {
    return Collections.unmodifiableMap(
        Maps.filterValues(
            Maps.transformValues(nodeMap, entry -> entry.isDone() ? entry.getValue() : null),
            Objects::nonNull));
  }

  @Override
  public Collection<InMemoryNodeEntry> getAllNodeEntries() {
    return Collections.unmodifiableCollection(nodeMap.values());
  }

  @Override
  public void parallelForEach(Consumer<InMemoryNodeEntry> consumer) {
    nodeMap.forEachValue(PARALLELISM_THRESHOLD, consumer);
  }

  @Override
  public void cleanupInterningPools() {
    if (!usePooledInterning) {
      return;
    }
    try (AutoProfiler ignored =
        GoogleAutoProfilerUtils.logged("Cleaning up interning pools", Duration.ofMillis(2L))) {
      parallelForEach(
          e -> {
            weakInternSkyKey(e.getKey());

            if (e.isDone() && e.getKey().functionName().equals(SkyFunctions.PACKAGE)) {
              weakInternPackageTargetsLabels((PackageValue) e.toValue());
            }

            // The graph is about to be thrown away. Remove as we go to avoid temporarily storing
            // everything in both the weak interner and the graph.
            nodeMap.remove(e.getKey());
          });
    }

    SkyKeyInterner.setGlobalPool(null);
    LabelInterner.setGlobalPool(null);
  }

  @Override
  @Nullable
  public InMemoryNodeEntry getIfPresent(SkyKey key) {
    return nodeMap.get(key);
  }

  static final class EdgelessInMemoryGraphImpl extends InMemoryGraphImpl {

    public EdgelessInMemoryGraphImpl(boolean usePooledInterning) {
      super(usePooledInterning);
    }

    @Override
    protected InMemoryNodeEntry newNodeEntry(SkyKey key) {
      return new NonIncrementalInMemoryNodeEntry(key);
    }
  }

  /** {@link PooledInterner.Pool} for {@link SkyKey}s. */
  final class SkyKeyPool implements PooledInterner.Pool<SkyKey> {

    @Override
    public SkyKey getOrWeakIntern(SkyKey sample) {
      // Use computeIfAbsent not to mutate the map, but to call weakIntern under synchronization.
      // This ensures that the canonical instance isn't being transferred to the node map
      // concurrently in createIfAbsent. In the common case that the key is already present in the
      // node map, this is a lock-free lookup.
      SkyKey[] weakInterned = new SkyKey[1];
      InMemoryNodeEntry nodeEntry =
          nodeMap.computeIfAbsent(
              sample,
              k -> {
                weakInterned[0] = k.getSkyKeyInterner().weakInternUnchecked(k);
                return null; // Don't actually store a mapping.
              });
      return nodeEntry != null ? nodeEntry.getKey() : weakInterned[0];
    }
  }

  /** {@link PooledInterner.Pool} for {@link Label}s. */
  final class LabelPool implements PooledInterner.Pool<Label> {
    @Override
    public Label getOrWeakIntern(Label sample) {
      LabelInterner interner = checkNotNull(Label.getLabelInterner());

      PackageIdentifier packageIdentifier = sample.getPackageIdentifier();

      // Return pooled instance if sample is present in the pool.
      InMemoryNodeEntry inMemoryNodeEntry = nodeMap.get(packageIdentifier);
      if (inMemoryNodeEntry != null) {
        Label pooledInstance = getLabelFromInMemoryNodeEntry(inMemoryNodeEntry, sample);
        if (pooledInstance != null) {
          return pooledInstance;
        }
      }

      Lock readLock = interner.getLockForLabelLookup(sample);
      readLock.lock();

      try {
        // Check again whether sample is already present in the pool inside critical section.
        if (inMemoryNodeEntry == null) {
          inMemoryNodeEntry = nodeMap.get(packageIdentifier);
        }

        if (inMemoryNodeEntry != null) {
          Label pooledInstance = getLabelFromInMemoryNodeEntry(inMemoryNodeEntry, sample);
          if (pooledInstance != null) {
            return pooledInstance;
          }
        }
        return interner.weakIntern(sample);
      } finally {
        readLock.unlock();
      }
    }
  }

  @Nullable
  private static Label getLabelFromInMemoryNodeEntry(
      InMemoryNodeEntry inMemoryNodeEntry, Label sample) {
    checkNotNull(inMemoryNodeEntry);
    SkyValue value = inMemoryNodeEntry.toValue();
    if (value == null) {
      return null;
    }
    checkState(value instanceof PackageValue, value);
    ImmutableSortedMap<String, Target> targets = ((PackageValue) value).getPackage().getTargets();
    Target target = targets.get(sample.getName());
    return target != null ? target.getLabel() : null;
  }
}
