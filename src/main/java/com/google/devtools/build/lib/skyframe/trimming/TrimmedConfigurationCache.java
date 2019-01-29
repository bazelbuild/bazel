// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.trimming;

import com.google.common.base.Preconditions;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Cache which tracks canonical invocations and matches keys to equivalent keys (after trimming).
 *
 * <p>This cache can be built independently of the massive build dependency that is build-base
 * (SkyFunctions and BuildConfiguration and so on), and so it is - thus, it uses type parameters to
 * speak more abstractly about what it cares about.
 *
 * <p>Consider a {@code <KeyT>} as a pair of {@code <DescriptorT>} and {@code <ConfigurationT>}. The
 * descriptor describes what the key builds, while the configuration describes how to build it.
 *
 * <p>For example, a ConfiguredTargetKey is made up of a Label, which is its descriptor, and a
 * BuildConfiguration, which is its configuration. An AspectKey is made up of a Label and a set of
 * AspectDescriptors describing the aspect and the aspects it depends on, all of which are part of
 * the AspectKey's descriptor, and also has a BuildConfiguration, which is its configuration.
 *
 * <p>A key always uses all of its descriptor, but it may only use part of its configuration. A Java
 * configured target may have no use for Python configuration, for example. Thus, it would produce
 * the same result to evaluate that target with a configuration which doesn't include Python data.
 * Reducing the configuration to the subset configuration which only includes the bits the target
 * actually needs is called trimming the configuration.
 *
 * <p>If this trimmed configuration is a subset of another configuration, then building whatever the
 * descriptor refers to with that other configuration will produce the same result as the trimmed
 * configuration, which is the same result as the configuration that the trimmed configuration was
 * trimmed from.
 *
 * <p>This cache provides methods for matching keys which would evaluate to the same result because
 * they have the same descriptor and trim to the same configuration, allowing callers to avoid doing
 * work that has already been done. It also permits invalidating, revalidating, and removing these
 * keys, as might happen during their lifecycle (if something they depend on has changed, etc.).
 *
 * <p>Internally, this cache is essentially a very sparse table. Each row, headed by a descriptor,
 * describes the possible configurations of that descriptor. Columns, headed by a trimmed
 * configuration, represent minimal configurations that descriptors can be invoked with. And a cell
 * contains the key which corresponds to the canonical invocation of that descriptor with that
 * configuration.
 *
 * <p>This class expects to be used in ways which are consistent with trimming. That is to say:
 *
 * <ul>
 *   <li>If the same key is put in the cache twice with different trimmed configurations, it must be
 *       invalidated between the two puts. Afterward, the original trimmed configuration is no
 *       longer valid for the rest of this build.
 *   <li>No trimmed configuration must be put in the cache which has equal values for every fragment
 *       it shares with another trimmed configuration already in the cache, unless the key
 *       associated with the other configuration has been invalidated. Afterward, the configuration
 *       which had previously been invalidated is no longer valid for the rest of this build.
 *   <li>Methods which read and add to the cache - {@link #get(KeyT)}, {@link #revalidate(KeyT)},
 *       and {@link #putIfAbsent(KeyT, ConfigurationT)} - may be used together in one phase of the
 *       build. Methods which remove from the cache - {@link #invalidate(KeyT)}, {@link
 *       #remove(KeyT)}, and {@link #clear()} - may be used together in another phase of the build.
 *       Calls to these groups of methods must never be interleaved.
 * </ul>
 *
 * <p>If used as described above, this class is thread-safe.
 */
public final class TrimmedConfigurationCache<KeyT, DescriptorT, ConfigurationT> {

  // ======== Tuning parameters ==========
  /** The initial capacity of the cache of descriptors. */
  private static final int CACHE_INITIAL_SIZE = 100;
  /** The table density for the cache of descriptors. */
  private static final float CACHE_LOAD_FACTOR = 0.9f;
  /** The number of threads expected to be writing to the descriptor cache at a time. */
  private static final int CACHE_CONCURRENCY_LEVEL = 16;
  /**
   * The number of configurations to expect in a single descriptor - that is, the initial capacity
   * of descriptors' maps.
   */
  private static final int EXPECTED_CONFIGURATIONS_PER_DESCRIPTOR = 4;
  /**
   * The table density for the {@link ConcurrentHashMap ConcurrentHashMaps} created for tracking
   * configurations of each descriptor.
   */
  private static final float DESCRIPTOR_LOAD_FACTOR = 0.9f;
  /** The number of threads expected to be writing to a single descriptor at a time. */
  private static final int DESCRIPTOR_CONCURRENCY_LEVEL = 1;

  private final Function<KeyT, DescriptorT> descriptorExtractor;
  private final Function<KeyT, ConfigurationT> configurationExtractor;

  private final ConfigurationComparer<ConfigurationT> configurationComparer;

  private volatile ConcurrentHashMap<
          DescriptorT, ConcurrentHashMap<ConfigurationT, KeyAndState<KeyT>>>
      descriptors;

  /**
   * Constructs a new TrimmedConfigurationCache with the given methods of extracting descriptors and
   * configurations from keys, and uses the given predicate to determine the relationship between
   * two configurations.
   *
   * <p>{@code configurationComparer} should be consistent with equals - that is,
   * {@code a.equals(b) == b.equals(a) == configurationComparer.compare(a, b).equals(Result.EQUAL)}
   */
  public TrimmedConfigurationCache(
      Function<KeyT, DescriptorT> descriptorExtractor,
      Function<KeyT, ConfigurationT> configurationExtractor,
      ConfigurationComparer<ConfigurationT> configurationComparer) {
    this.descriptorExtractor = descriptorExtractor;
    this.configurationExtractor = configurationExtractor;
    this.configurationComparer = configurationComparer;
    this.descriptors = newCacheMap();
  }

  /**
   * Looks for a key with the same descriptor as the input key, which has a configuration that
   * trimmed to a subset of the input key's.
   *
   * <p>Note that this is not referring to a <em>proper</em> subset; it's quite possible for a key
   * to "trim" to a configuration equal to its configuration. That is, without anything being
   * removed.
   *
   * <p>If such a key has been added to this cache, it is returned in a present {@link Optional}.
   * Invoking this key will produce the same result as invoking the input key.
   *
   * <p>If no such key has been added to this cache, or if a key has been added to the cache and
   * subsequently been the subject of an {@link #invalidate(KeyT)}, an absent Optional will be
   * returned instead. No currently-valid key has trimmed to an equivalent configuration, and so the
   * input key should be executed.
   */
  public Optional<KeyT> get(KeyT input) {
    DescriptorT descriptor = getDescriptorFor(input);
    ConcurrentHashMap<ConfigurationT, KeyAndState<KeyT>> trimmingsOfDescriptor =
        descriptors.get(descriptor);
    if (trimmingsOfDescriptor == null) {
      // There are no entries at all for this descriptor.
      return Optional.empty();
    }
    ConfigurationT candidateConfiguration = getConfigurationFor(input);
    for (Entry<ConfigurationT, KeyAndState<KeyT>> entry : trimmingsOfDescriptor.entrySet()) {
      ConfigurationT trimmedConfig = entry.getKey();
      KeyAndState<KeyT> canonicalKeyAndState = entry.getValue();
      if (canSubstituteFor(candidateConfiguration, trimmedConfig, canonicalKeyAndState)) {
        return Optional.of(canonicalKeyAndState.getKey());
      }
    }
    return Optional.empty();
  }

  /**
   * Returns whether the given trimmed configuration and key are a suitable substitute for the
   * candidate configuration.
   */
  private boolean canSubstituteFor(
      ConfigurationT candidateConfiguration,
      ConfigurationT trimmedConfiguration,
      KeyAndState<KeyT> canonicalKeyAndState) {
    return canonicalKeyAndState.getState().isKnownValid()
        && compareConfigurations(trimmedConfiguration, candidateConfiguration).isSubsetOrEqual();
  }

  /**
   * Attempts to record the given key as the canonical invocation for its descriptor and the
   * passed-in trimmed configuration.
   *
   * <p>The trimmed configuration must be a subset of the input key's configuration. Otherwise,
   * {@link IllegalArgumentException} will be thrown.
   *
   * <p>If another key matching this configuration is found, that key will be returned. That key
   * represents the canonical invocation, which should produce the same result as the input key. It
   * may have been previously invalidated, but will be considered revalidated at this point.
   *
   * <p>Otherwise, if the input key is the first to trim to this configuration, the input key is
   * returned.
   */
  public KeyT putIfAbsent(KeyT canonicalKey, ConfigurationT trimmedConfiguration) {
    ConfigurationT fullConfiguration = getConfigurationFor(canonicalKey);
    Preconditions.checkArgument(
        compareConfigurations(trimmedConfiguration, fullConfiguration).isSubsetOrEqual());
    ConcurrentHashMap<ConfigurationT, KeyAndState<KeyT>> trimmingsOfDescriptor =
        descriptors.computeIfAbsent(getDescriptorFor(canonicalKey), unused -> newDescriptorMap());
    KeyAndState<KeyT> currentMapping =
        trimmingsOfDescriptor.compute(
            trimmedConfiguration,
            (configuration, currentValue) -> {
              if (currentValue == null) {
                return KeyAndState.create(canonicalKey);
              } else {
                return currentValue.asValidated();
              }
            });
    boolean newlyAdded = currentMapping.getKey().equals(canonicalKey);
    int failedRemoves;
    do {
      failedRemoves = 0;
      for (Entry<ConfigurationT, KeyAndState<KeyT>> entry : trimmingsOfDescriptor.entrySet()) {
        if (entry.getValue().getState().equals(KeyAndState.State.POSSIBLY_INVALID)) {
          // Remove invalidated keys where:
          // * the same key evaluated to a different configuration than it does now
          // * (for trimmed configurations not yet seen) the new trimmed configuration has equal
          //   values for every fragment it shares with the old configuration (including subsets
          //   or supersets).
          // These are keys we know will not be revalidated as part of the current build.
          // Although it also ensures that we don't remove the entry we just added, the check for
          // invalidation is mainly to avoid wasting time checking entries that are still valid for
          // the current build and therefore will not match either of these properties.
          if (entry.getValue().getKey().equals(canonicalKey)
              || (newlyAdded
                  && compareConfigurations(trimmedConfiguration, entry.getKey())
                      .hasEqualSharedFragments())) {
            if (!trimmingsOfDescriptor.remove(entry.getKey(), entry.getValue())) {
              // It's possible that this entry was removed by another thread in the meantime.
              failedRemoves += 1;
            }
          }
        }
      }
    } while (failedRemoves > 0);
    return currentMapping.getKey();
  }

  /**
   * Marks the given key as invalidated.
   *
   * <p>An invalidated key will not be returned from {@link #get(KeyT)}, as it cannot be proven that
   * the key will still trim to the same configuration.
   *
   * <p>This invalidation is undone if the input key is passed to {@link #revalidate(KeyT)}, or if
   * the configuration it originally trimmed to is passed to a call of {@link putIfAbsent(KeyT,
   * ConfigurationT)}. This is true regardless of whether the key passed to putIfAbsent is the same
   * as the input to this method.
   *
   * <p>If the key is not currently canonical for any descriptor/configuration pair, or if the key
   * had previously been invalidated and not revalidated, this method has no effect.
   */
  public void invalidate(KeyT key) {
    updateEntryWithRetries(key, KeyAndState::asInvalidated);
  }

  /**
   * Unmarks the given key as invalidated.
   *
   * <p>This undoes the effects of {@link #invalidate(KeyT)}, allowing the key to be returned from
   * {@link #get(KeyT)} again.
   *
   * <p>If the key is not currently canonical for any descriptor/configuration pair, or if the key
   * had not previously been invalidated or had since been revalidated, this method has no effect.
   */
  public void revalidate(KeyT key) {
    updateEntryWithRetries(key, KeyAndState::asValidated);
  }

  /**
   * Completely removes the given key from the cache.
   *
   * <p>After this call, {@link #get(KeyT)} and {@link #putIfAbsent(KeyT, ConfigurationT)} will no
   * longer return this key unless it is put back in the cache with putIfAbsent.
   *
   * <p>If the key is not currently canonical for any descriptor/configuration pair, this method has
   * no effect.
   */
  public void remove(KeyT key) {
    // Return null from the transformer to remove the key from the map.
    updateEntryWithRetries(key, unused -> null);

    DescriptorT descriptor = getDescriptorFor(key);
    ConcurrentHashMap<ConfigurationT, KeyAndState<KeyT>> trimmingsOfDescriptor =
        descriptors.get(descriptor);
    if (trimmingsOfDescriptor != null && trimmingsOfDescriptor.isEmpty()) {
      descriptors.remove(descriptor, trimmingsOfDescriptor);
    }
  }

  /**
   * Finds the entry in the cache where the given key is canonical and updates or removes it.
   *
   * <p>The transformation is applied transactionally; that is, if another change has happened since
   * the value was first looked up, the new value is retrieved and the transformation is applied
   * again. This repeats until there are no conflicts.
   *
   * <p>This method has no effect if this key is currently not canonical.
   *
   * @param transformation The transformation to apply to the given entry. The entry will be
   *     replaced with the value returned from invoking this on the original value. If it returns
   *     null, the entry will be removed instead. If it returns the same instance, nothing will be
   *     done to the entry.
   */
  private void updateEntryWithRetries(KeyT key, UnaryOperator<KeyAndState<KeyT>> transformation) {
    while (!updateEntryIfNoConflicts(key, transformation)) {}
  }

  /**
   * Finds the entry in the cache where the given key is canonical and updates or removes it.
   *
   * <p>Only one attempt is made, and if there's a collision with another change, false is returned
   * and the map is not changed.
   *
   * <p>This method succeeds (returns {@code true}) without doing anything if this key is currently
   * not canonical.
   *
   * @param transformation The transformation to apply to the given entry. The entry will be
   *     replaced with the value returned from invoking this on the original value. If it returns
   *     null, the entry will be removed instead. If it returns the same instance, nothing will be
   *     done to the entry.
   */
  private boolean updateEntryIfNoConflicts(
      KeyT key, UnaryOperator<KeyAndState<KeyT>> transformation) {
    DescriptorT descriptor = getDescriptorFor(key);
    ConcurrentHashMap<ConfigurationT, KeyAndState<KeyT>> trimmingsOfDescriptor =
        descriptors.get(descriptor);
    if (trimmingsOfDescriptor == null) {
      // There are no entries at all for this descriptor.
      return true;
    }

    for (Entry<ConfigurationT, KeyAndState<KeyT>> entry : trimmingsOfDescriptor.entrySet()) {
      KeyAndState<KeyT> currentValue = entry.getValue();
      if (currentValue.getKey().equals(key)) {
        KeyAndState<KeyT> newValue = transformation.apply(currentValue);
        if (newValue == null) {
          return trimmingsOfDescriptor.remove(entry.getKey(), currentValue);
        } else if (newValue != currentValue) {
          return trimmingsOfDescriptor.replace(entry.getKey(), currentValue, newValue);
        } else {
          // newValue == currentValue, there's nothing to do
          return true;
        }
      }
    }
    // The key requested wasn't in the map, so there's nothing to do
    return true;
  }

  /**
   * Removes all keys from this cache, resetting it to its empty state.
   *
   * <p>This is equivalent to calling {@link #remove(KeyT)} on every key which had ever been passed
   * to {@link #putIfAbsent(KeyT, ConfigurationT)}.
   */
  public void clear() {
    // Getting a brand new instance lets the old map be garbage collected, reducing its memory
    // footprint from its previous expansions.
    this.descriptors = newCacheMap();
  }

  /** Retrieves the descriptor by calling the descriptorExtractor. */
  private DescriptorT getDescriptorFor(KeyT key) {
    return descriptorExtractor.apply(key);
  }

  /** Retrieves the configuration by calling the configurationExtractor. */
  private ConfigurationT getConfigurationFor(KeyT key) {
    return configurationExtractor.apply(key);
  }

  /**
   * Checks whether the first configuration is equal to or a subset of the second by calling the
   * configurationComparer.
   */
  private ConfigurationComparer.Result compareConfigurations(
      ConfigurationT left, ConfigurationT right) {
    return configurationComparer.apply(left, right);
  }

  /** Generates a new map suitable for storing the cache as a whole. */
  private ConcurrentHashMap<DescriptorT, ConcurrentHashMap<ConfigurationT, KeyAndState<KeyT>>>
      newCacheMap() {
    return new ConcurrentHashMap<>(CACHE_INITIAL_SIZE, CACHE_LOAD_FACTOR, CACHE_CONCURRENCY_LEVEL);
  }

  /** Generates a new map suitable for storing the cache of configurations for a descriptor. */
  private ConcurrentHashMap<ConfigurationT, KeyAndState<KeyT>> newDescriptorMap() {
    return new ConcurrentHashMap<>(
        EXPECTED_CONFIGURATIONS_PER_DESCRIPTOR,
        DESCRIPTOR_LOAD_FACTOR,
        DESCRIPTOR_CONCURRENCY_LEVEL);
  }
}
