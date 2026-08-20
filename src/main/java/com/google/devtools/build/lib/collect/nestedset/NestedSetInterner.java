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

package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.base.Preconditions.checkNotNull;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.common.base.Equivalence;
import com.google.devtools.build.lib.runtime.MemoryOptimizations;
import java.util.Arrays;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkValue;

/**
 * Interner for {@link NestedSet} children arrays.
 *
 * <p>We intern only the backing {@code Object[]} children arrays rather than the enclosing {@link
 * NestedSet} instances. {@link Depset} equality and hashing delegate directly to the underlying
 * {@link NestedSet} instance's identity. If {@link NestedSet} instances were interned in a bounded
 * cache, depset equality/hash comparisons in Starlark would become non-deterministic based on cache
 * hits versus evictions (b/544770151). Interning only the backing arrays achieves most of the
 * memory savings while maintaining deterministic instance identity.
 */
public final class NestedSetInterner {
  private final Cache<Equivalence.Wrapper<Object[]>, Object[]> cache;
  private final Equivalence<Object[]> equivalence;

  private NestedSetInterner(int size, Equivalence<Object[]> equivalence) {
    this.cache = Caffeine.newBuilder().maximumSize(size).build();
    this.equivalence = equivalence;
  }

  private Object[] internImpl(Object[] sample) {
    checkNotNull(sample);
    return cache.get(equivalence.wrap(sample), Equivalence.Wrapper::get);
  }

  private void clearImpl() {
    cache.invalidateAll();
  }

  // Use a duo of strong interners to dedupe some NestedSet children arrays at low cost. For most
  // builds this saves memory, and for some builds this saves CPU and wall time too (due to fewer
  // duplicate expansions, due to getting sharing via #cached).
  //
  // We use equality-based interning for a few types where it both makes sense and profiling showed
  // there were sufficient duplicates for interning to be worth it.
  //
  // We use identity-based interning for Starlark values (such as Artifact and other types
  // wrapped by Depsets, with the exception of the above types). Profiling showed this was worth it.
  //
  // Motivation for identity-based interning of NestedSet<Artifact>: Artifacts get mutated via
  // setGeneratingActionKey during ConfiguredTargetFunction. This is problematic for equality-based
  // global interning of NestedSet<Artifact>. If a children array from before a CTF restart for the
  // node is used on the subsequent CTF computation for the node, then we would have an
  // inconsistency due to setGeneratingActionKey being called on the new Artifact instance, but the
  // original Artifact instance in the interned array actually getting used. There are various ways
  // to work around this, but for the sake of simplicity and low cost, we went with identity-based
  // interning for StarlarkValue types, which uses a separate interner.
  private static final NestedSetInterner identityInterner =
      new NestedSetInterner(
          /* size= */ 25000,
          new Equivalence<>() {
            @Override
            @SuppressWarnings("ReferenceEquality") // Cares about identity of children elements.
            protected boolean doEquivalent(Object[] arrayA, Object[] arrayB) {
              if (arrayA == arrayB) {
                return true;
              }
              if (arrayA.length != arrayB.length) {
                return false;
              }
              for (int i = 0; i < arrayA.length; i++) {
                if (arrayA[i] != arrayB[i]) {
                  return false;
                }
              }
              return true;
            }

            @Override
            protected int doHash(Object[] array) {
              int result = 0;
              for (Object element : array) {
                result = 31 * result + System.identityHashCode(element);
              }
              return result;
            }
          });
  private static final NestedSetInterner equalityInterner =
      new NestedSetInterner(
          /* size= */ 10000,
          new Equivalence<>() {
            @Override
            protected boolean doEquivalent(Object[] arrayA, Object[] arrayB) {
              return Arrays.equals(arrayA, arrayB);
            }

            @Override
            protected int doHash(Object[] array) {
              return Arrays.hashCode(array);
            }
          });

  private static boolean enabled() {
    // Since we use bounded caches, the efficacy of nested set interning is non-deterministic.
    // In addition, interning nested sets may lead to non-deterministic serialization.
    // Therefore we respect both MemoryOptimizations knobs.
    return MemoryOptimizations.allowNonDeterministicEfficacy.get()
        && MemoryOptimizations.allowNonDeterministicSerialization.get();
  }

  /** Interns a {@link NestedSet} children array. */
  static Object[] intern(Object[] children) {
    if (!enabled()) {
      return children;
    }
    NestedSetInterner interner = getInterner(children);
    if (interner == null) {
      return children;
    }
    return interner.internImpl(children);
  }

  public static void clear() {
    identityInterner.clearImpl();
    equalityInterner.clearImpl();
  }

  @Nullable
  private static NestedSetInterner getInterner(Object[] array) {
    // Pick the interner, if any, to use based on a representative element of the nested set.
    while (array.length > 0 && array[0] instanceof Object[]) {
      array = (Object[]) array[0];
    }
    if (array.length == 0) {
      // Should never happen in practice via calls from NestedSet.create().
      return null;
    }
    return getInterner(array[0].getClass());
  }

  @Nullable
  private static NestedSetInterner getInterner(Class<?> elementClass) {
    if (String.class.isAssignableFrom(elementClass)
        || NestedSetsShouldBeInternedByEquality.class.isAssignableFrom(elementClass)) {
      return equalityInterner;
    }
    if (StarlarkValue.class.isAssignableFrom(elementClass)) {
      return identityInterner;
    }
    return null;
  }
}
