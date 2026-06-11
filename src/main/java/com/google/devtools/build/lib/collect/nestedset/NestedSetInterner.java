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
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/** Interner for {@link NestedSet} instances. */
public class NestedSetInterner {
  private static final AtomicBoolean enabled = new AtomicBoolean(true);

  private final Cache<Equivalence.Wrapper<NestedSet<?>>, NestedSet<?>> cache;
  private final Equivalence<NestedSet<?>> equivalence;

  private NestedSetInterner(int size, Equivalence<NestedSet<?>> equivalence) {
    this.cache = Caffeine.newBuilder().maximumSize(size).build();
    this.equivalence = equivalence;
  }

  private NestedSet<?> internImpl(NestedSet<?> sample) {
    checkNotNull(sample);
    return cache.get(equivalence.wrap(sample), Equivalence.Wrapper::get);
  }

  private void clearImpl() {
    cache.invalidateAll();
  }

  // Use a duo of strong interners to dedupe some NestedSet instances at low cost. For most builds
  // this saves 3-5% memory, and for some builds this saves CPU and wall time too (due to fewer
  // duplicate expansions, due to getting sharing via #cached).
  //
  // We used equality-based interning for a few types where it both makes sense and profiling showed
  // there were sufficient duplicates for interning to be worth it.
  //
  // We use identity-based interning for all NestedSets wrapped by Depsets (with the except of the
  // above types), and also for NestedSet<Artifact>. Profiling showed this was worth it.
  //
  // Motivation for identity-based interning of NestedSet<Artifact>: Artifacts get mutated via
  // setGeneratingActionKey during ConfiguredTargetFunction. This is problematic for equality-based
  // global interning of NestedSet<Artifact>. If a NestedSet<Artifact> instance from before a CTF
  // restart for the node is used on the subsequent CTF computation for the node, then we're going
  // to have an inconsistency due to setGeneratingActionKey being called on the new Artifact
  // instance, but the original NestedSet<Artifact> instance actually getting used. There are
  // various ways to work around this, but for the sake of simplicity and low cost, we went with the
  // approach of identity-based interning for NestedSet<Artifact>, which means we want a separate
  // interner.
  private static final NestedSetInterner identityInterner =
      new NestedSetInterner(
          /* size= */ 25000,
          new Equivalence<NestedSet<?>>() {
            @Override
            @SuppressWarnings("ReferenceEquality") // Cares about identity of children elements.
            protected boolean doEquivalent(NestedSet<?> nestedSetA, NestedSet<?> nestedSetB) {
              if (nestedSetA.getOrder() != nestedSetB.getOrder()) {
                return false;
              }
              if (nestedSetA.isSingleton() && nestedSetB.isSingleton()) {
                return nestedSetA.children == nestedSetB.children;
              }
              if (nestedSetA.isSingleton() || nestedSetB.isSingleton()) {
                return false;
              }
              if (nestedSetA.children instanceof Object[] arrayA
                  && nestedSetB.children instanceof Object[] arrayB) {
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
              return false;
            }

            @Override
            protected int doHash(NestedSet<?> nestedSet) {
              int result = nestedSet.getOrder().hashCode();
              if (nestedSet.isSingleton()) {
                return 31 * result + System.identityHashCode(nestedSet.children);
              }
              if (nestedSet.children instanceof Object[] array) {
                for (Object element : array) {
                  result = 31 * result + System.identityHashCode(element);
                }
                return result;
              }
              throw new IllegalStateException(
                  "Unexpected children type: " + nestedSet.children.getClass());
            }
          });
  private static final NestedSetInterner equalityInterner =
      new NestedSetInterner(
          /* size= */ 10000,
          new Equivalence<NestedSet<?>>() {
            @Override
            protected boolean doEquivalent(NestedSet<?> nestedSetA, NestedSet<?> nestedSetB) {
              return nestedSetA.shallowEquals(nestedSetB);
            }

            @Override
            protected int doHash(NestedSet<?> nestedSet) {
              return nestedSet.shallowHashCode();
            }
          });

  /** Enables or disables interning. */
  // TODO: b/522370193 - Figure out if this is actually needed.
  public static void setEnabled(boolean enabled) {
    NestedSetInterner.enabled.set(enabled);
  }

  /** Interns a {@link NestedSet} instance. */
  public static <T> NestedSet<T> intern(NestedSet<T> nestedSet) {
    if (!enabled.get()) {
      return nestedSet;
    }
    NestedSetInterner interner = getInterner(nestedSet);
    if (interner == null) {
      return nestedSet;
    }
    @SuppressWarnings("unchecked") // Practically, a hit will always have the same element type.
    NestedSet<T> interned = (NestedSet<T>) interner.internImpl(nestedSet);
    return interned;
  }

  /** Interns a {@link NestedSet} instance wrapped by a {@link Depset}. */
  public static NestedSet<?> internDepset(NestedSet<?> nestedSet, Class<?> elementClass) {
    if (!enabled.get()) {
      return nestedSet;
    }
    NestedSetInterner interner = getInterner(elementClass);
    if (interner == null) {
      // Always intern NestedSets for Depsets, at least by identity. In practice there are lots of
      // duplicate Depsets, even with respect to identity.
      interner = identityInterner;
    }
    return interner.internImpl(nestedSet);
  }

  public static void clear() {
    identityInterner.clearImpl();
    equalityInterner.clearImpl();
  }

  @Nullable
  private static NestedSetInterner getInterner(NestedSet<?> nestedSet) {
    Object representative = nestedSet.children;
    // Pick the interner, if any, to use based on a representative element of the nested set.
    if (representative instanceof Object[] array) {
      // This loop doesn't show up in profiles. If it ever does, we can consider another approach to
      // picking the interner to use.
      while (array.length > 0 && array[0] instanceof Object[]) {
        array = (Object[]) array[0];
      }
      if (array.length == 0) {
        // Should never happen in practice via calls from NestedSetBuilder.build().
        return null;
      }
      representative = array[0];
    }
    return getInterner(representative.getClass());
  }

  @Nullable
  private static NestedSetInterner getInterner(Class<?> elementClass) {
    if (String.class.isAssignableFrom(elementClass)
        || NestedSetsShouldBeInternedByEquality.class.isAssignableFrom(elementClass)) {
      return equalityInterner;
    }
    if (IsArtifactForNestedSet.class.isAssignableFrom(elementClass)) {
      return identityInterner;
    }
    return null;
  }
}
