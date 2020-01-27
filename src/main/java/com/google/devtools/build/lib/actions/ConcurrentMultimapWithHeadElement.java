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
package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;
import java.util.Iterator;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/**
 * A Multimap-like object that is actually a {@link ConcurrentMap} of {@code SmallSet}s to avoid
 * the memory penalties of a {@code Multimap} while preserving concurrency guarantees, and
 * retrieving a consistent "head" element. Operations are guaranteed to reflect a consistent view of
 * a {@code SetMultimap}, although most methods are not implemented.
 */
final class ConcurrentMultimapWithHeadElement<K, V> {
  private final ConcurrentMap<K, SmallSet<V>> map = Maps.newConcurrentMap();

  /**
   * Remove (key, val) pair from the multimap. If this removes the current 'head' element
   * for a key, then another randomly chosen element becomes the current head.
   *
   * <p>Until the next (possibly concurrent) {@link #putAndGet}(key, val) call, {@link #get}(key)
   * will never return val.
   */
  void remove(K key, V val) {
    SmallSet<V> entry = getEntry(key);
    if (entry != null) {
      entry.remove(val);
      if (entry.get() == null) {
        // Remove entry completely from map if dead.
        map.remove(key, entry);
      }
    }
  }

  /**
   * Return some value val such that (key, val) is in the multimap. If there is always at least one
   * entry for key in the multimap during the lifetime of this method call, it will not return null.
   */
  @Nullable V get(K key) {
    SmallSet<V> entry = getEntry(key);
    return (entry != null) ? entry.get() : null;
  }

  /**
   * Adds (key, val) to the multimap. Returns the head element for key, either val or another
   * already-stored value.
   */
  V putAndGet(K key, V val) {
    V result = null;
    while (result == null) {
      // If another thread concurrently removes the only remaining value from the entry, this
      // putAndGet will return null, since the entry is about to be removed from the map. In that
      // case, we obtain a fresh entry from the map and do the put on it.
      result = getOrCreateEntry(key).putAndGet(val);
    }
    return result;
  }

  /**
   * Obtain the entry for key, adding it to the underlying map if no entry was previously present.
   */
  private SmallSet<V> getOrCreateEntry(K key) {
    SmallSet<V> entry = new SmallSet<V>();
    SmallSet<V> oldEntry = map.putIfAbsent(key, entry);
    if (oldEntry != null) {
      return oldEntry;
    }
    return entry;
  }

  /**
   * Obtain the entry for key, returning null if no entry was present in the underlying map.
   */
  private SmallSet<V> getEntry(K key) {
    return map.get(key);
  }

  /**
   * Clears the multimap. May not be called concurrently with any other methods.
   */
  @ThreadHostile
  void clear() {
    map.clear();
  }

  /**
   * Wrapper for a {@code #Set} that will probably have at most one element. Keeps the first element
   * in a separate variable for fast reading/writing and to save space if more than one element is
   * never written to this set. We always have the invariant that {@link #first} is null only if
   * {@link #rest} is null.
   */
  private static class SmallSet<T> {
    /*
     * What is this 'volatile' on first and where's the lock on the read path?
     *
     * Volatile is an alternative to locking that works only in very limited situations, such as
     * simple field reads and writes.  Writes from one thread to 'first' happen before reads from
     * other threads.  When used correctly, it can have the same correctness properties as a
     * 'synchronized' but is much faster on most hardware.
     *
     * Here, volatile is used to eliminate locks on the read path.  Since get() is merely fetching
     * the contents of 'first', it meets the criteria for a safe volatile read.  In the mutator
     * methods, care is taken to write only correct values to 'first'; intermediate and incomplete
     * values do not get written to the field.  This means that whenever 'first' is replaced, it is
     * immediately replaced with the next correct value.  Therefore, it is a safe volatile write.
     *
     * Other more complex relationships that need to be maintained during the mutate are maintained
     * with the Object monitor.  Since they do not impact the read path (only 'first' matters), the
     * lock is sufficient for writes and unnecessary for 'first' reads.
     *
     * Documentation on volatile:
     * http://docs.oracle.com/javase/7/docs/api/java/util/concurrent/package-summary.html#MemoryVisibility
     * (java.util.concurrent package docs)
     */

    private volatile T first = null;
    private Set<T> rest = null;

    /*
     * We may have a race where one thread tries to remove a small set from the map while another
     * thread tries to add to it. If the second thread loses the race, it will add to a set that is
     * no longer in the map. To prevent that, once a small set is ever empty, we mark it "dead" by
     * setting {@code rest} to a {@code TOMBSTONE} value, and (and subsequently remove it from the
     * map). No modifications to a set can happen after the {@code TOMBSTONE} value is set. Thus,
     * the thread trying to add a new value to a set will fail, and knows to retrieve the entry anew
     * from the map and try again.
     */
    private static final ImmutableSet<Object> TOMBSTONE = ImmutableSet.of();

    /**
     * Return some value in the SmallSet.
     *
     * <p>If there is always at least one value in the SmallSet during the lifetime of this call,
     * it will not return null, since by the invariant, {@link #first} must be non-null.
     */
    private T get() {
      return first;
    }

    /**
     * Adds val to the SmallSet. Returns some element of the SmallSet.
     */
    private synchronized T putAndGet(T elt) {
      Preconditions.checkNotNull(elt);
      if (isDead()) {
        return null;
      }
      if (elt.equals(first)) {
        return first;
      }
      if (first == null) {
        Preconditions.checkState(rest == null, elt);
        first = elt;
        return first;
      }
      if (rest == null) {
        rest = Sets.newHashSet();
      }
      rest.add(elt);
      return first;
    }

    /**
     * Remove val from the SmallSet, if it is present.
     */
    private synchronized void remove(T elt) {
      Preconditions.checkNotNull(elt);
      if (isDead()) {
        return;
      }
      if (elt.equals(first)) {
        // Normalize to enforce invariant "first is null only if rest is empty."
        if (rest != null) {
          Iterator<T> it = rest.iterator();
          first = it.next();
          it.remove();
          if (!it.hasNext()) {
            rest = null;
          }
        } else {
          first = null;
          markDead();
        }
      } else if ((rest != null) && rest.remove(elt) && rest.isEmpty()) { // side-effect: remove
        rest = null;
      }
    }

    private boolean isDead() {
      Preconditions.checkState(rest != TOMBSTONE || first == null,
          "%s present in tombstoned SmallSet, but tombstoned SmallSets should be empty", first);
      return rest == TOMBSTONE;
    }

    @SuppressWarnings("unchecked") // Cast of TOMBSTONE. Ok since TOMBSTONE is empty immutable set.
    private void markDead() {
      rest = (Set<T>) TOMBSTONE;
    }
  }
}
